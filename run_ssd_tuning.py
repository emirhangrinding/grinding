#!/usr/bin/env python3
import random
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST, CIFAR10
import argparse

from utils import set_global_seed, SEED_DEFAULT
from data import generate_subdatasets, MultiTaskDataset, transform_mnist, transform_test_cifar, create_subset_data_loaders
from models import MTL_Two_Heads_ResNet
from tuning import optimise_ssd_hyperparams

# Configuration - EDIT THESE VALUES
# MODEL_PATH = "baseline_mtl_all_clients.h5" 
DATASET_NAME = "CIFAR10"
TARGET_SUBSET_ID = 0
NUM_CLIENTS = 10
BATCH_SIZE = 128
N_TRIALS = 100
DATA_ROOT = "./data"
HEAD_SIZE = "medium"
SEED = 42

# Setup
parser = argparse.ArgumentParser(description="Run SSD tuning for an MTL ResNet model.")
parser.add_argument("--model-path", type=str, required=True, help="Path to the baseline MTL model file.")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_global_seed(SEED)

# Generate data (FIXED - removed seed parameter)
clients_data, clients_labels, full_dataset = generate_subdatasets(
    dataset_name=DATASET_NAME,
    setting="non-iid",
    num_clients=NUM_CLIENTS,
    data_root=DATA_ROOT
)

# Create datasets and loaders
mtl_dataset = MultiTaskDataset(full_dataset, clients_data)
dataset_size = len(mtl_dataset)
train_size = int(0.8 * dataset_size)
indices = list(range(dataset_size))
# Note: Shuffling is removed to be consistent with finetune_after_ssd.py
# random.shuffle(indices)
train_indices = indices[:train_size]

train_dataset = Subset(mtl_dataset, train_indices)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
retain_loader, forget_loader = create_subset_data_loaders(train_loader, TARGET_SUBSET_ID)

# Test loader
if DATASET_NAME == "MNIST":
    test_base = MNIST(root=DATA_ROOT, train=False, download=True, transform=transform_mnist)
else:
    test_base = CIFAR10(root=DATA_ROOT, train=False, download=True, transform=transform_test_cifar)

test_class_indices = {i: [] for i in range(10)}
for idx, (_, label) in enumerate(test_base):
    test_class_indices[label].append(idx)

test_clients_data = {k: [] for k in clients_data.keys()}
for class_label, idxs in test_class_indices.items():
    client_ids = [cid for cid in clients_data.keys() 
                 if any(full_dataset[i][1] == class_label for i in clients_data[cid])]
    if not client_ids:
        client_ids = list(clients_data.keys())
    for i, idx in enumerate(idxs):
        client = client_ids[i % len(client_ids)]
        test_clients_data[client].append(idx)

test_mtl_dataset = MultiTaskDataset(test_base, test_clients_data)
test_loader = DataLoader(test_mtl_dataset, batch_size=BATCH_SIZE)

# Load model
model = MTL_Two_Heads_ResNet(dataset_name=DATASET_NAME, num_clients=NUM_CLIENTS, head_size=HEAD_SIZE)
model.load_state_dict(torch.load(args.model_path, map_location=device))
model.to(device)
model.eval()

# Run optimization
study = optimise_ssd_hyperparams(
    pretrained_model=model,
    retain_loader=retain_loader,
    forget_loader=forget_loader,
    test_loader=test_loader,
    device=device,
    target_subset_id=TARGET_SUBSET_ID,
    n_trials=N_TRIALS,
    seed=SEED
)

print(f"Best α: {study.best_params['alpha']:.6f}")
print(f"Best λ: {study.best_params['lambda']:.6f}")