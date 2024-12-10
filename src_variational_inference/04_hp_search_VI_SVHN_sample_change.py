import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import pathlib
import optuna
import numpy as np
import random
import json
from torchvision import datasets

# Set random seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# Data Preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))  # SVHN-specific stats
])

trainset = datasets.SVHN(root="./data", split="train", transform=transform, download=True)
testset = datasets.SVHN(root="./data", split="test", transform=transform, download=True)

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

    def forward(self, x):
        batch_size = x.size(0)

        weight_sigma = F.softplus(self.weight_rho)
        bias_sigma = F.softplus(self.bias_rho)

        r_w = torch.randn_like(self.weight_mu)
        r_b = torch.randn_like(self.bias_mu)

        weight = self.weight_mu + r_w * weight_sigma
        bias = self.bias_mu + r_b * bias_sigma

        mean_output = F.linear(x, weight, bias)

        # Generate input and output signs
        input_sign = torch.bernoulli(torch.full((batch_size, self.in_features), 0.5, device=x.device)).mul(2).sub(1)
        output_sign = torch.bernoulli(torch.full((batch_size, self.out_features), 0.5, device=x.device)).mul(2).sub(1)

        perturbation = F.linear(x * input_sign, weight) * output_sign
        return mean_output + perturbation

    def kl_loss(self):
        weight_sigma = F.softplus(self.weight_rho)
        bias_sigma = F.softplus(self.bias_rho)

        kl_weight = 0.5 * torch.sum(
            torch.log1p((weight_sigma ** 2) / (self.prior_sigma ** 2)) +
            (self.weight_mu ** 2) / (self.prior_sigma ** 2) - 1
        )
        kl_bias = 0.5 * torch.sum(
            torch.log1p((bias_sigma ** 2) / (self.prior_sigma ** 2)) +
            (self.bias_mu ** 2) / (self.prior_sigma ** 2) - 1
        )
        return kl_weight + kl_bias

trainset, valset = create_validation_set(trainset, 500)

class BayesianNNFlipout(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

        # Dynamically compute flattened size
        self.flattened_size = self._get_flattened_size(3, 32)

        self.fc1 = FlipoutLinear(self.flattened_size, 512, prior_sigma=0.1)
        self.fc2 = FlipoutLinear(512, 10, prior_sigma=0.1)

    def _get_flattened_size(self, channels, dim):
        with torch.no_grad():
            x = torch.zeros(1, channels, dim, dim)
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            return x.view(-1).size(0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, self.flattened_size)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

    def kl_loss(self):
        return self.fc1.kl_loss() + self.fc2.kl_loss()


# Objective Function for Optuna
def objective(trial, size, trainset, valset, device, results_dir):
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    prior_sigma = trial.suggest_float("prior_sigma", 0.01, 1.0, log=True)
    learning_rate = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    epochs = trial.suggest_int("epochs", 50, 100)

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

    # Initialize the model
    model = BayesianNNFlipout().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(epochs):
        model.train()
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            nll_loss = F.cross_entropy(outputs, labels)
            kl_loss = model.kl_loss()
            beta = min(1.0, epoch / 10)  # Warm-up for KL divergence
            loss = nll_loss + beta * kl_loss
            loss.backward()
            optimizer.step()

    # Evaluate the model
    model.eval()
    val_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in valloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            val_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
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

# Hyperparameter Tuning
#sample_sizes = [1, 4000, 2000, 500, 100, 50]
sample_sizes = [50, 100, 500, 2000, 4000]
device = torch.device("mps" if torch.cuda.is_available() else "cpu")


for size in sample_sizes:
    completed_trials = []
    results_dir = f"./results/VI_HP_{size}"
    pathlib.Path(results_dir).mkdir(parents=True, exist_ok=True)
    results_path = f"{results_dir}/optuna_results_sample_{size}.json"

    print(f"\nStarting hyperparameter search for VI with {size} samples per class...")
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, size, trainset, valset, device, results_dir), n_trials=20)

    # Save completed trials
    trials_data = [{"params": t.params, "value": t.value} for t in study.trials]
    completed_trials.append({"size": size, "trials": trials_data})

    with open(results_path, "w") as f:
        json.dump(completed_trials, f, indent=4)

    best_trial = study.best_trial
    print(f"Best trial for VI with {size} samples: {best_trial.params}")

    with open(f"{results_dir}/best_params_sample_{size}.txt", "w") as f:
        f.write("Best parameters found:\n")
        f.write(json.dumps(best_trial.params, indent=4))
        f.write("\n")
        f.write(f"Best trial number: {best_trial.number}")

    print(f"Saved best model configuration to {results_dir}/best_params_sample_{size}.txt")