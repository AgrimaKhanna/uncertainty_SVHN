import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score
import numpy as np
import json
import pathlib

# Set random seeds
torch.manual_seed(0)
np.random.seed(0)

# Evaluate the model with MCDO and calculate entropy
def mc_dropout_predict_with_entropy(model, inputs, n_samples=10):
    model.train()  # Enable dropout during prediction
    outputs = torch.stack([model(inputs) for _ in range(n_samples)])
    mean_output = outputs.mean(dim=0)
    uncertainty = outputs.var(dim=0)
    # Calculate entropy
    probs = torch.softmax(mean_output, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=1)  # Add epsilon to avoid log(0)
    max_probs, _ = torch.max(probs, dim=1)  # Get max probability
    return mean_output, uncertainty, entropy, max_probs


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
        # Dense layers
        self.fc1 = nn.Linear(conv_neurons * 4 * 4, num_neurons)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(num_neurons, 10)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.view(x.size(0), -1)
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
        x = x.view(x.size(0), -1)
        x = self.dropout(self.base_model.relu3(self.base_model.fc1(x)))
        x = self.base_model.fc2(x)
        return x


# Data preparation
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize to match input size
    transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

ind_loader = DataLoader(torchvision.datasets.SVHN(root='./data', train=False, download=True, transform=transform),
                        batch_size=64, shuffle=False)
ood_loader = DataLoader(torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform),
                        batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameter loading function
def load_hyperparameters(hp_path):
    with open(hp_path, "r") as f:
        lines = f.readlines()
    batch_size = int(lines[2].split(":")[1].strip().strip(","))
    num_neurons = int(lines[3].split(":")[1].strip().strip(","))
    conv_neurons = int(lines[4].split(":")[1].strip().strip(","))
    num_layers = int(lines[5].split(":")[1].strip().strip(","))
    drop_out = float(lines[6].split(":")[1].strip().strip(","))
    lr = float(lines[7].split(":")[1].strip().strip(","))
    num_epochs = int(lines[8].split(":")[1].strip().strip(","))
    return {
        'batch_size': batch_size,
        'num_neurons': num_neurons,
        'conv_neurons': conv_neurons,
        'num_layers': num_layers,
        'drop_out': drop_out,
        'lr': lr,
        'num_epochs': num_epochs
    }

# Evaluation for different sample sizes
sample_sizes = [1, 5, 10, 50, 100, 2000, 4000]

for size in sample_sizes:
    if size == 100 or size == 50:
        hp_size = 4000
    else:
        hp_size = size

    hp_path = f"../src_MCDO/hp_results/MCDO_HP_size_{hp_size}/best_params_sample_{hp_size}.txt"
    hyperparameters = load_hyperparameters(hp_path)

    print(f"Hyperparameters for {size} samples: {hyperparameters}")

    # Initialize model
    base_model = SimpleCNN(
        kernel_size=3,
        num_neurons=hyperparameters['num_neurons'],
        conv_neurons=hyperparameters['conv_neurons'],
        num_layers=hyperparameters['num_layers']
    ).to(device)
    model = SimpleCNNWithMCDO(base_model, drop_out=hyperparameters['drop_out']).to(device)

    # Load pre-trained model
    model.load_state_dict(torch.load(f"../src/models/svhn_model_{size}_samples_mcdo.pth"))

    all_entropy_ind = []
    all_entropy_ood = []
    all_max_prob_ind = []
    all_max_prob_ood = []

    # Evaluate in-distribution
    with torch.no_grad():
        for inputs, _ in ind_loader:
            inputs = inputs.to(device)
            _, _, entropy, max_probs = mc_dropout_predict_with_entropy(model, inputs, n_samples=10)
            all_entropy_ind.extend(entropy.cpu().numpy())
            all_max_prob_ind.extend(max_probs.cpu().numpy())

    # Evaluate out-of-distribution
    with torch.no_grad():
        for inputs, _ in ood_loader:
            inputs = inputs.to(device)
            _, _, entropy, max_probs = mc_dropout_predict_with_entropy(model, inputs, n_samples=10)
            all_entropy_ood.extend(entropy.cpu().numpy())
            all_max_prob_ood.extend(max_probs.cpu().numpy())

    # Compute OOD AUC for entropy
    labels = np.array([0] * len(all_entropy_ind) + [1] * len(all_entropy_ood))
    scores_entropy = np.array(all_entropy_ind + all_entropy_ood)
    auc_entropy = roc_auc_score(labels, scores_entropy)

    # Compute OOD AUC for max probabilities
    scores_max_prob = np.array(all_max_prob_ind + all_max_prob_ood)
    auc_max_prob = roc_auc_score(labels, -scores_max_prob)  # Negate for inverse relation

    # Compute mean and std for entropy and max probabilities
    results = {
        "auc_entropy": auc_entropy,
        "auc_max_prob": auc_max_prob,
        "entropy": {
            "ind_mean": np.mean(all_entropy_ind),
            "ind_std": np.std(all_entropy_ind),
            "ood_mean": np.mean(all_entropy_ood),
            "ood_std": np.std(all_entropy_ood),
        },
        "max_prob": {
            "ind_mean": np.mean(all_max_prob_ind),
            "ind_std": np.std(all_max_prob_ind),
            "ood_mean": np.mean(all_max_prob_ood),
            "ood_std": np.std(all_max_prob_ood),
        }
    }

    # Save results
    results_path = f"./results/mcdo/svhn_ood_results_{size}_samples.json"
    pathlib.Path("./results/mcdo/").mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved at {results_path}")
