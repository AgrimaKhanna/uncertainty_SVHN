# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, Subset
import pathlib
import random
import numpy as np
import json
import torch.nn.functional as F
from torchvision import datasets, transforms

# Setting the seed
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# %%
# Improved data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))  # SVHN specific stats
])

train_dataset = datasets.SVHN(root='./data', split='train', transform=transform, download=True)
test_dataset = datasets.SVHN(root='./data', split='test', transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

class FlipoutLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_sigma=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma
        
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features).normal_(0, 0.01))
        self.weight_rho = nn.Parameter(torch.zeros(out_features, in_features).normal_(-5, 0.01))
        self.bias_mu = nn.Parameter(torch.zeros(out_features).normal_(0, 0.01))
        self.bias_rho = nn.Parameter(torch.zeros(out_features).normal_(-5, 0.01))
        
        # Random signs for flipout
        self.register_buffer('input_sign', None)
        self.register_buffer('output_sign', None)
        
    def get_random_signs(self, batch_size, shape, device):
        return (2 * torch.bernoulli(torch.ones(batch_size, *shape, device=device) * 0.5) - 1)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        if (self.input_sign is None or 
            self.input_sign.size(0) != batch_size or 
            self.output_sign is None or 
            self.output_sign.size(0) != batch_size):
            
            self.input_sign = self.get_random_signs(batch_size, (self.in_features,), x.device)
            self.output_sign = self.get_random_signs(batch_size, (self.out_features,), x.device)
        
        weight_sigma = F.softplus(self.weight_rho)
        bias_sigma = F.softplus(self.bias_rho)
        
        mean_output = F.linear(x, self.weight_mu, self.bias_mu)
        
        # Compute perturbation with corrected dimensions
        r_w = torch.randn_like(self.weight_mu)
        perturbed_weights = (r_w * weight_sigma).unsqueeze(0)
        
        x_reshape = x * self.input_sign
        perturbation = F.linear(x_reshape, perturbed_weights.squeeze(0)) * self.output_sign
        
        bias_perturbation = bias_sigma * torch.randn_like(self.bias_mu)
        
        return mean_output + perturbation + bias_perturbation

    def kl_loss(self):
        weight_sigma = F.softplus(self.weight_rho)
        bias_sigma = F.softplus(self.bias_rho)
        
        kl_weight = 0.5 * torch.sum(
            torch.log1p((weight_sigma**2) / (self.prior_sigma**2)) +
            (self.weight_mu**2) / (self.prior_sigma**2) - 1
        )
        
        kl_bias = 0.5 * torch.sum(
            torch.log1p((bias_sigma**2) / (self.prior_sigma**2)) +
            (self.bias_mu**2) / (self.prior_sigma**2) - 1
        )
        
        return (kl_weight + kl_bias) / self.weight_mu.numel()

class BayesianNNFlipout(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = FlipoutLinear(64 * 4 * 4, 512, prior_sigma=0.1)
        self.fc2 = FlipoutLinear(512, 10, prior_sigma=0.1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 4 * 4)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

    def kl_loss(self):
        return self.fc1.kl_loss() + self.fc2.kl_loss()

def train_epoch(model, train_loader, optimizer, device, epoch, total_epochs=100):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    beta = min(1.0, (epoch / (total_epochs * 0.2))) * 0.1
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        output = model(data)
        nll_loss = F.cross_entropy(output, target)
        kl_loss = model.kl_loss()
        loss = nll_loss + beta * kl_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        if batch_idx % 100 == 0:
            print(f'Train Batch: {batch_idx}/{len(train_loader)}, '
                  f'Loss: {loss.item():.4f}, '
                  f'Acc: {100. * correct/total:.2f}%')
    
    return total_loss / len(train_loader), correct / total

def evaluate(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            test_loss += F.cross_entropy(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = correct / len(test_loader.dataset)
    
    print(f'Test set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {100. * accuracy:.2f}%')
    
    return test_loss, accuracy


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
sample_sizes = [1, 50, 100, 500, 2000, 4000]
device = torch.device("mps" if torch.cuda.is_available() else "cpu")

# %%
for size in sample_sizes:
    hp_size = size

    hp_path = f"./results/MCDO_HP_size_{hp_size}/best_params_sample_{hp_size}.txt"

    # Open the file and extract hyperparameters
    with open(hp_path, "r") as f:
        lines = f.readlines()

    # Parse the hyperparameters
    batch_size = int(lines[2].split(":")[1].strip().strip(","))
    weight_decay = int(lines[3].split(":")[1].strip().strip(","))
    lr = int(lines[4].split(":")[1].strip().strip(","))
    beta_warmup_epochs = int(lines[5].split(":")[1].strip().strip(","))
    epochs = float(lines[6].split(":")[1].strip().strip(","))
    best_trial = int(lines[10].split(":")[1].strip())

    print(f"Best hyperparameters for {size} samples: {batch_size}, {weight_decay}, {lr}, {beta_warmup_epochs}, {epochs}, {best_trial}")

    hyperparameters = {
        'lr': lr, 
        'num_epochs': epochs, 
        'batch_size': batch_size, 
        'weight_decay': weight_decay, 
        'beta_warmup_epochs': beta_warmup_epochs, 
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

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = BayesianNNFlipout().to(device)
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters['lr'], weight_decay=hyperparameters['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # Training loop with beta warmup
    n_epochs = hyperparameters['num_epochs']
    beta_warmup_epochs = hyperparameters['beta_warmup_epochs']

    for epoch in range(n_epochs):
        beta = min(1.0, epoch / beta_warmup_epochs)  # Linear warmup of KL term
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, beta)
        test_loss, test_acc = evaluate(model, test_loader, device)
        
        scheduler.step(test_loss)
        
        print(f'Epoch: {epoch+1}/{n_epochs}')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {100*train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {100*test_acc:.2f}%')
        print('-' * 60)

    # Save the model
    model_dir = "./models/"
    pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)
    model_path = f"{model_dir}/svhn_model_{size}_samples_VI.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}")

    # Evaluate the model with VI
    def VI_predict(model, inputs, n_samples=10):
        model.train()  
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
            mean_output, uncertainty = VI_predict(model, inputs, n_samples=10)
            _, predicted = torch.max(mean_output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_uncertainties.append(uncertainty.cpu().numpy())

    accuracy = 100 * correct / total
    print(f"Accuracy on test set: {accuracy:.2f}%")

    # Save the results
    results_path = f"./results/svhn_results_{size}_samples_VI.json"
    pathlib.Path("./results/").mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({"accuracy": accuracy}, f)
    print(f"Results saved at {results_path}")


