#!/usr/bin/env python3
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST, CIFAR10

from utils import set_global_seed, SEED_DEFAULT
from data import generate_subdatasets, transform_mnist, transform_test_cifar
from models import StandardResNet
from tuning import optimise_ssd_hyperparams

# Configuration - EDIT THESE VALUES
MODEL_PATH = "baseline_all_clients_model.h5"  # From train_baseline_all_no_mtl.py
DATASET_NAME = "CIFAR10"
TARGET_SUBSET_ID = 0
NUM_CLIENTS = 10
BATCH_SIZE = 128
N_TRIALS = 100
DATA_ROOT = "./data"
SEED = 42

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_global_seed(SEED)

# Generate data
clients_data, _, full_dataset = generate_subdatasets(
    dataset_name=DATASET_NAME,
    setting="non-iid",
    num_clients=NUM_CLIENTS,
    data_root=DATA_ROOT
)

# Create retain and forget loaders
target_client_key = f"client{TARGET_SUBSET_ID + 1}"
forget_indices = clients_data[target_client_key]

retain_indices = []
for k, v in clients_data.items():
    if k != target_client_key:
        retain_indices.extend(v)

forget_dataset = Subset(full_dataset, forget_indices)
retain_dataset = Subset(full_dataset, retain_indices)

retain_loader = DataLoader(retain_dataset, batch_size=BATCH_SIZE, shuffle=True)
forget_loader = DataLoader(forget_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Test loader
if DATASET_NAME == "MNIST":
    test_base = MNIST(root=DATA_ROOT, train=False, download=True, transform=transform_mnist)
else:
    test_base = CIFAR10(root=DATA_ROOT, train=False, download=True, transform=transform_test_cifar)

test_loader = DataLoader(test_base, batch_size=BATCH_SIZE)

# Load model
model = StandardResNet(dataset_name=DATASET_NAME)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Run optimization
study = optimise_ssd_hyperparams(
    pretrained_model=model,
    retain_loader=retain_loader,
    forget_loader=forget_loader,
    test_loader=test_loader,
    device=device,
    target_subset_id=None,  # Not needed for single-head model
    n_trials=N_TRIALS,
    seed=SEED
)

print(f"Best α: {study.best_params['alpha']:.6f}")
print(f"Best λ: {study.best_params['lambda']:.6f}") 