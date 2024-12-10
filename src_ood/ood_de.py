import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np
import json
import pathlib
import pandas as pd

# Set random seeds
torch.manual_seed(0)
np.random.seed(0)

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

        # Flatten and dense layers
        self.fc1 = nn.Linear(conv_neurons * 4 * 4, num_neurons)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(num_neurons, 10)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


# Define the Deep Ensemble Model
class DeepEnsemble:
    def __init__(self, base_model_class, num_models, device, *model_args, **model_kwargs):
        self.models = [base_model_class(*model_args, **model_kwargs).to(device) for _ in range(num_models)]

    def predict(self, inputs):
        inputs = inputs.to(next(self.models[0].parameters()).device)
        predictions = [torch.softmax(model(inputs), dim=1) for model in self.models]
        return torch.mean(torch.stack(predictions), dim=0)


# Data Preparation
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize to match CNN input
    transforms.Grayscale(num_output_channels=3),  # Convert to RGB
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load datasets
ind_loader = DataLoader(torchvision.datasets.SVHN(root='./data', train=False, download=True, transform=transform),
                        batch_size=64, shuffle=False)
ood_loader = DataLoader(torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform),
                        batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
ensemble_size = 5
sample_sizes = [1, 5, 10, 50, 100, 500, 2000, 4000]

# Load hyperparameters from CSV
hp_csv_path = '../src_deep_ensemble/hp_results/DE_HP_4000/results_4000_top_5.csv'
hp_df = pd.read_csv(hp_csv_path)
hp_df.reset_index(drop=True, inplace=True)

# Define function to calculate entropy
def calculate_entropy(probs):
    return -torch.sum(probs * torch.log(probs + 1e-12), dim=1)

# Evaluate OOD metrics
for size in sample_sizes:
    # Get hyperparameters for the current sample size
    row = hp_df.iloc[0]  # Use the top hyperparameter set
    hyperparameters = {
        'batch_size': int(row['batch_size']),
        'num_neurons': int(row['num_neurons']),
        'conv_neurons': int(row['conv_neurons']),
        'num_layers': int(row['num_layers']),
        'lr': row['lr'],
        'epochs': int(row['epochs']),
    }

    print(f"Loading models for sample size {size} with hyperparameters: {hyperparameters}")

    # Initialize the Deep Ensemble
    ensemble = DeepEnsemble(
        SimpleCNN,
        ensemble_size,
        device,
        kernel_size=3,
        num_neurons=hyperparameters['num_neurons'],
        conv_neurons=hyperparameters['conv_neurons'],
        num_layers=hyperparameters['num_layers']
    )

    # Load pre-trained models
    model_dir = "../src_deep_ensemble/models"
    for i in range(ensemble_size):
        model_path = f"{model_dir}/svhn_model_{size}_samples_de_{i + 1}.pth"  # Use SVHN models
        try:
            ensemble.models[i].load_state_dict(torch.load(model_path, map_location=device))  # Map to current device
        except FileNotFoundError:
            print(f"Model not found: {model_path}")
            continue

    all_entropy_ind = []
    all_entropy_ood = []
    all_max_prob_ind = []
    all_max_prob_ood = []

    # Evaluate in-distribution
    with torch.no_grad():
        for inputs, _ in ind_loader:
            inputs = inputs.to(device)
            outputs = ensemble.predict(inputs)

            entropy = calculate_entropy(outputs)
            all_entropy_ind.extend(entropy.cpu().numpy())

            max_probs, _ = torch.max(outputs, dim=1)
            all_max_prob_ind.extend(max_probs.cpu().numpy())

    # Evaluate out-of-distribution
    with torch.no_grad():
        for inputs, _ in ood_loader:
            inputs = inputs.to(device)
            outputs = ensemble.predict(inputs)

            entropy = calculate_entropy(outputs)
            all_entropy_ood.extend(entropy.cpu().numpy())

            max_probs, _ = torch.max(outputs, dim=1)
            all_max_prob_ood.extend(max_probs.cpu().numpy())

    # Compute OOD AUC for entropy
    labels = np.array([0] * len(all_entropy_ind) + [1] * len(all_entropy_ood))
    scores_entropy = np.array(all_entropy_ind + all_entropy_ood)
    auc_entropy = roc_auc_score(labels, scores_entropy)

    # Compute OOD AUC for max probabilities
    scores_max_prob = np.array(all_max_prob_ind + all_max_prob_ood)
    auc_max_prob = roc_auc_score(labels, -scores_max_prob)  # Negate for inverse relation

    # Compute mean and std for entropy and max probabilities
    mean_entropy_ind = np.mean(all_entropy_ind)
    std_entropy_ind = np.std(all_entropy_ind)
    mean_entropy_ood = np.mean(all_entropy_ood)
    std_entropy_ood = np.std(all_entropy_ood)

    mean_max_prob_ind = np.mean(all_max_prob_ind)
    std_max_prob_ind = np.std(all_max_prob_ind)
    mean_max_prob_ood = np.mean(all_max_prob_ood)
    std_max_prob_ood = np.std(all_max_prob_ood)

    # Save results
    results = {
        "auc_entropy": auc_entropy,
        "auc_max_prob": auc_max_prob,
        "entropy": {
            "ind_mean": mean_entropy_ind,
            "ind_std": std_entropy_ind,
            "ood_mean": mean_entropy_ood,
            "ood_std": std_entropy_ood,
        },
        "max_prob": {
            "ind_mean": mean_max_prob_ind,
            "ind_std": std_max_prob_ind,
            "ood_mean": mean_max_prob_ood,
            "ood_std": std_max_prob_ood,
        }
    }

    output_path = f"./results/mnist_ood_results_{size}_samples.json"
    pathlib.Path("./results/").mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved at {output_path}")
