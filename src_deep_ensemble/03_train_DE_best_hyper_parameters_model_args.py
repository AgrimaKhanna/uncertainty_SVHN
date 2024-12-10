# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split
import pathlib
import json
import numpy as np
import random
import pandas as pd
# Setting the seed
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

import argparse

# Get the model number from the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model_number", help="The model number to train", type=int)
args = parser.parse_args()

model_number = int(args.model_number)


# %%
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

# Deep Ensemble Model
# class DeepEnsemble:
#     def __init__(self, base_model_class, num_models, *model_args, **model_kwargs):
#         self.models = [base_model_class(*model_args, **model_kwargs) for _ in range(num_models)]


class DeepEnsemble:
    def __init__(self, base_model_class, num_models, device, *model_args, **model_kwargs):
        self.models = [base_model_class(*model_args, **model_kwargs).to(device) for _ in range(num_models)]

    def train(self, trainloader, criterion, optimizers, epochs, device):
        for epoch in range(epochs):
            for model, optimizer in zip(self.models, optimizers):
                model.train()
                for inputs, labels in trainloader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

            print(f"Epoch {epoch + 1}/{epochs} completed!")

    def predict(self, inputs):
        inputs = inputs.to(next(self.models[0].parameters()).device)  # Move inputs to the same device as the model
        predictions = [torch.softmax(model(inputs), dim=1) for model in self.models]
        return torch.mean(torch.stack(predictions), dim=0)
# %%

# Data preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform)
testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# Create validation set with 500 samples per class
def create_validation_set(trainset, samples_per_class):
    indices = []
    class_counts = {i: 0 for i in range(10)}
    for idx, (_, label) in enumerate(trainset):
        if class_counts[label] < samples_per_class:
            indices.append(idx)
            class_counts[label] += 1
        if all(count >= samples_per_class for count in class_counts.values()):
            break
    val_subset = Subset(trainset, indices)
    remaining_indices = [idx for idx in range(len(trainset)) if idx not in indices]
    train_subset = Subset(trainset, remaining_indices)
    return train_subset, val_subset

# %%

# Training and evaluation for different sample sizes
sample_sizes = [1, 5, 10, 50, 100, 500, 2000, 4000]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
ensemble_size = 1

for size in sample_sizes:

    hp_df = pd.read_csv('./results/DE_HP_4000/results_4000_top_5.csv')
    hp_df.reset_index(drop=True, inplace=True)
    # Get the second best hyperparameters
    row = hp_df.iloc[model_number-1]

    hyerparameters = {
        'batch_size': int(row['batch_size']),
        'num_neurons': int(row['num_neurons']),
        'conv_neurons': int(row['conv_neurons']),
        'num_layers': int(row['num_layers']),
        'lr': row['lr'],
        'epochs': int(row['epochs']),
    }

    print(hyerparameters)

    print(f"\nTraining model with {size} samples per class model {model_number}...")
    
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
    trainloader = DataLoader(subset, batch_size=64, shuffle=True)

    # Initialize model, loss, and optimizer
    base_model_class = SimpleCNN
    ensemble = DeepEnsemble(base_model_class, ensemble_size, device, kernel_size=3, num_neurons=hyerparameters['num_neurons'], conv_neurons=hyerparameters['conv_neurons'], num_layers=hyerparameters['num_layers'])
    # def __init__(self, base_model_class, num_models, device, *model_args, **model_kwargs):

    criterion = nn.CrossEntropyLoss()
    optimizers = [optim.Adam(model.parameters(), lr=hyerparameters['lr']) for model in ensemble.models]

    # Train the model
    ensemble.train(trainloader, criterion, optimizers, epochs=hyerparameters['epochs'], device=device)

    ensemble.models = [model.to(device) for model in ensemble.models]


    # Save the model
    model_dir = "./models_each_model/"
    pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)

    ens_count = 1
    for model in ensemble.models:
        model_path = f"{model_dir}/svhn_model_{size}_samples_de_{ens_count}_model_{model_number}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved at {model_path}")
        ens_count += 1

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = ensemble.predict(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy of the ensemble on the test set: {accuracy:.2f}%")

    # Save the results
    results_path = f"./results_per_model/svhn_results_{size}_samples_model_{model_number}.json"
    pathlib.Path("./results_per_model/").mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({"accuracy": accuracy}, f)
    print(f"Results saved at {results_path}")


