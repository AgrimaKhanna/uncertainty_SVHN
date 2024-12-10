# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import pathlib
import random
import numpy as np
import json
# Setting the seed
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import pathlib
import random

# Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self, kernel_size=3, num_neurons=128, conv_neurons=32, num_layers=2):
        super(SimpleCNN, self).__init__()
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Conv2d(3, conv_neurons, kernel_size=kernel_size, stride=1, padding=1))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # Intermediate layers
        for _ in range(num_layers - 1):
            self.layers.append(nn.Conv2d(conv_neurons, conv_neurons * 2, kernel_size=kernel_size, stride=1, padding=1))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            conv_neurons *= 2

        # Calculate the size of the flattened output
        self.flattened_size = self._get_flattened_size(3, 32, kernel_size, num_layers)

        self.fc1 = nn.Linear(self.flattened_size, num_neurons)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(num_neurons, 10)

    def _get_flattened_size(self, input_channels, input_dim, kernel_size, num_layers):
        """Dynamically calculate the flattened size after conv and pooling layers."""
        x = torch.zeros((1, input_channels, input_dim, input_dim))
        for layer in self.layers:
            x = layer(x)
        return x.numel()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the MCDO wrapper for uncertainty estimation
class SimpleCNNWithMCDO(nn.Module):
    def __init__(self, base_model, drop_out=0.5):
        super(SimpleCNNWithMCDO, self).__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x):
        for layer in self.base_model.layers:
            x = self.dropout(layer(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(self.base_model.relu3(self.base_model.fc1(x)))
        x = self.base_model.fc2(x)
        return x


# %%

# Data preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform)
testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# %%

# Training and evaluation for different sample sizes
sample_sizes = [500, 2000, 4000]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
for size in sample_sizes:
    if size == 100 or size == 50:
        hp_size = 4000
    else:
        hp_size = size

    hp_path = f"./results/MCDO_HP_size_{hp_size}/best_params_sample_{hp_size}.txt"

    # Open the file and extract hyperparameters
    with open(hp_path, "r") as f:
        lines = f.readlines()

    # Parse the hyperparameters
    batch_size = int(lines[2].split(":")[1].strip().strip(","))
    num_neurons = int(lines[3].split(":")[1].strip().strip(","))
    conv_neurons = int(lines[4].split(":")[1].strip().strip(","))
    num_layers = int(lines[5].split(":")[1].strip().strip(","))
    drop_out = float(lines[6].split(":")[1].strip().strip(","))
    lr = float(lines[7].split(":")[1].strip().strip(","))
    num_epochs = int(lines[8].split(":")[1].strip().strip(","))
    best_trial = int(lines[10].split(":")[1].strip())

    print(f"Best hyperparameters for {size} samples: {batch_size}, {num_neurons}, {conv_neurons}, {num_layers}, {drop_out}, {lr}, {num_epochs}")

    hyperparameters = {
        'lr': lr, 
        'drop_out': drop_out, 
        'num_epochs': num_epochs, 
        'batch_size': batch_size, 
        'num_neurons': num_neurons, 
        'conv_neurons': conv_neurons, 
        'num_layers': num_layers, 
    }

    print(f"\nTraining model with {size} samples per class...")
    
    # Subset dataset to include only 'size' samples per class
    indices = []
    class_counts = {i: 0 for i in range(10)}
    for idx, (_, label) in enumerate(trainset):
        if class_counts[label] < size:
            indices.append(idx)
            class_counts[label] += 1
        if all(count >= size for count in class_counts.values()):
            break

    subset = Subset(trainset, indices)
    trainloader = DataLoader(subset, batch_size=hyperparameters['batch_size'], shuffle=True)

    # Initialize model, loss, and optimizer
    base_model = SimpleCNN(kernel_size=3, num_neurons=hyperparameters['num_neurons'], conv_neurons=hyperparameters['conv_neurons'], num_layers=hyperparameters['num_layers']).to(device) 
    model = SimpleCNNWithMCDO(base_model, drop_out=hyperparameters['drop_out']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(base_model.parameters(), lr=hyperparameters['lr'])

    # Train the model
    num_epochs = hyperparameters['num_epochs']
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(trainloader):.3f}")

    # Save the model
    model_dir = "./models/"
    pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)
    model_path = f"{model_dir}/svhn_model_{size}_samples_mcdo.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}")

    # Evaluate the model with MCDO
    def mc_dropout_predict(model, inputs, n_samples=10):
        model.train()  # Enable dropout during prediction
        outputs = torch.stack([model(inputs) for _ in range(n_samples)])
        mean_output = outputs.mean(dim=0)
        uncertainty = outputs.var(dim=0)
        return mean_output, uncertainty

    model.eval()
    correct = 0
    total = 0
    all_uncertainties = []
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            mean_output, uncertainty = mc_dropout_predict(model, inputs, n_samples=10)
            _, predicted = torch.max(mean_output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_uncertainties.append(uncertainty.cpu().numpy())

    accuracy = 100 * correct / total
    print(f"Accuracy on test set: {accuracy:.2f}%")

    # Save the results
    results_path = f"./results/svhn_results_{size}_samples_mcdo.json"
    pathlib.Path("./results/").mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({"accuracy": accuracy}, f)
    print(f"Results saved at {results_path}")


