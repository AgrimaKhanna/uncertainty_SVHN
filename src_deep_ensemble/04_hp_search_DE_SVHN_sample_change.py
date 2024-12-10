import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split
import pathlib
import optuna
import numpy as np 
import random
import json 

# Set random seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

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
class DeepEnsemble:
    def __init__(self, base_model_class, num_models, *model_args, **model_kwargs):
        self.models = [base_model_class(*model_args, **model_kwargs) for _ in range(num_models)]

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

    def predict(self, inputs):
        predictions = [torch.softmax(model(inputs), dim=1) for model in self.models]
        return torch.mean(torch.stack(predictions), dim=0)

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

trainset, valset = create_validation_set(trainset, 500)

# Objective function for Optuna hyperparameter tuning
def objective(trial, size, trainset, valset, device, results_dir):
    kernel_size = 3
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    num_neurons = trial.suggest_categorical('num_neurons', [64, 128, 256])
    conv_neurons = trial.suggest_categorical('conv_neurons', [16, 32, 64])
    num_layers = trial.suggest_int('num_layers', 2, 4)
    learning_rate = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    epochs = trial.suggest_int('epochs', 50, 100)

    # Subset dataset for the given sample size
    indices = []
    class_counts = {i: 0 for i in range(10)}
    for idx, (_, label) in enumerate(trainset):
        if class_counts[label] < size:
            indices.append(idx)
            class_counts[label] += 1
        if all(count >= size for count in class_counts.values()):
            break

    subset = Subset(trainset, indices)
    trainloader = DataLoader(subset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    # Initialize ensemble
    ensemble_size = 5
    base_model_class = SimpleCNN
    ensemble = DeepEnsemble(base_model_class, ensemble_size, kernel_size=kernel_size, num_neurons=num_neurons, conv_neurons=conv_neurons, num_layers=num_layers)
    optimizers = [optim.Adam(model.parameters(), lr=learning_rate) for model in ensemble.models]

    # Train ensemble
    ensemble.train(trainloader, nn.CrossEntropyLoss(), optimizers, epochs, device)

    # Evaluate on validation set
    ensemble.models = [model.to(device) for model in ensemble.models]
    val_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in valloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = ensemble.predict(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            val_loss += loss.item()

            predicted = torch.argmax(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    # Save current trial's results
    trial_result = {
        "params": trial.params,
        "value": val_loss / len(valloader),
        "accuracy": accuracy
    }
    with open(f"{results_dir}/result_trial_{trial.number}.json", "a") as f:
        f.write(json.dumps(trial_result) + "\n")

    return val_loss / len(valloader)

# Hyperparameter tuning for each sample size
sample_sizes = [500, 50, 10]

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

for size in sample_sizes:
    completed_trials = []
    results_dir = f"./results/DE_HP_{size}"
    pathlib.Path(results_dir).mkdir(parents=True, exist_ok=True)
    results_path = f"{results_dir}/optuna_results_sample_{size}.json"

    print(f"\nStarting hyperparameter search for {size} samples per class...")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, size, trainset, valset, device, results_dir), n_trials=20, n_jobs=5)


    # Save completed trials
    trials_data = [{"params": t.params, "value": t.value} for t in study.trials]
    completed_trials.append({"size": size, "trials": trials_data})

    with open(results_path, "w") as f:
        json.dump(completed_trials, f, indent=4)

    best_trial = study.best_trial
    print(f"Best trial for {size} samples: {best_trial.params}")

    with open(f"{results_dir}/best_params_sample_{size}.txt", "w") as f:
        f.write("Best parameters found:\n")
        f.write(json.dumps(best_trial.params, indent=4))
        # Add the best trial number as well to the file 
        f.write("\n")
        f.write(f"Best trial number: {best_trial.number}")

    print(f"Saved best model configuration to {results_dir}/best_params_sample_{size}.txt")
