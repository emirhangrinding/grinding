#!/usr/bin/env python3
import random
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
import argparse

from utils import set_global_seed, SEED_DEFAULT
from data import generate_subdatasets, MultiTaskDataset, transform_mnist, transform_test_cifar, create_subset_data_loaders
from models import MTL_Two_Heads_ResNet
from tuning import optimise_ssd_hyperparams

# Configuration
DATASET_NAME = "CIFAR100"
TARGET_SUBSET_ID = 0
NUM_CLIENTS = 10
BATCH_SIZE = 64
N_TRIALS = 50
DATA_ROOT = "./data"
HEAD_SIZE = "medium"
SEED = 42

# Setup
parser = argparse.ArgumentParser(description="Run SSD tuning for an MTL ResNet model.")
parser.add_argument("--model-path", type=str, required=True, help="Path to the baseline MTL model file.")
parser.add_argument("--target-subset-id", type=int, default=TARGET_SUBSET_ID, help="The ID of the client to forget.")
parser.add_argument("--num-forgotten-clients", type=int, default=1, help="The number of clients that have been forgotten so far (including the current one).")
parser.add_argument("--unlearned-model-name", type=str, default="unlearned_model_mtl", help="Name for the output unlearned model file.")
parser.add_argument("--previous-forgotten-clients", type=int, nargs='*', default=[], help="List of client IDs that were forgotten in previous rounds.")
parser.add_argument("--fisher-on", type=str, choices=["subset", "digit"], default="subset", help="Task to compute Fisher Information on: 'subset' or 'digit'.")
parser.add_argument("--kill-output-neuron", action="store_true", help="If set, suppress the target subset's output neuron during evaluation after SSD.")
parser.add_argument("--digit-metrics-only", action="store_true", help="If set, Optuna will optimize using only digit accuracies (ignores subset-ID metrics).")
parser.add_argument("--baseline-variant", type=str, choices=["mtl", "mtl_ce", "no_mtl"], default=None, help="Baseline set to use for sequential forgetting metrics.")
parser.add_argument("--current-client-id", type=int, default=None, help="Explicit client id to use for per-round baselines (defaults to target-subset-id).")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_global_seed(SEED)

# Generate data
clients_data, _, full_dataset = generate_subdatasets(
    dataset_name=DATASET_NAME,
    setting="non-iid",
    num_clients=NUM_CLIENTS,
    data_root=DATA_ROOT
)

"""Build MultiTaskDataset and map original dataset indices to its index space."""
mtl_dataset = MultiTaskDataset(full_dataset, clients_data)
dsidx_to_mtlidx = {ds_idx: pos for pos, ds_idx in enumerate(mtl_dataset.valid_indices)}

# Exclude previously forgotten clients from the training pool for SSD tuning
prev_forgotten_set = set(args.previous_forgotten_clients or [])
train_mtl_indices = []
for c_id_str, ds_indices in clients_data.items():
    numeric_id = int(c_id_str.replace("client", "")) - 1
    if numeric_id not in prev_forgotten_set:
        train_mtl_indices.extend(
            [dsidx_to_mtlidx[i] for i in ds_indices if i in dsidx_to_mtlidx]
        )

train_dataset = Subset(mtl_dataset, train_mtl_indices)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Create loaders for retain and the CURRENTLY forgotten client (split within the filtered pool)
retain_loader, forget_loader = create_subset_data_loaders(train_loader, args.target_subset_id)

# Create loaders for PREVIOUSLY forgotten clients
all_forgotten_loaders = {}
if args.previous_forgotten_clients:
    print(f"Creating loaders for previously forgotten clients: {args.previous_forgotten_clients}")
    for client_id in args.previous_forgotten_clients:
        # We need to create a loader for each previously forgotten client.
        # This requires isolating their data from the full dataset.
        client_indices = clients_data[f"client{client_id + 1}"]
        client_mtl_indices = [dsidx_to_mtlidx[i] for i in client_indices if i in dsidx_to_mtlidx]
        forget_dataset = Subset(mtl_dataset, client_mtl_indices)
        all_forgotten_loaders[client_id] = DataLoader(forget_dataset, batch_size=BATCH_SIZE, shuffle=False)

"""Test loader selection based on dataset."""
if DATASET_NAME == "MNIST":
    test_base = MNIST(root=DATA_ROOT, train=False, download=True, transform=transform_mnist)
elif DATASET_NAME == "CIFAR10":
    test_base = CIFAR10(root=DATA_ROOT, train=False, download=True, transform=transform_test_cifar)
else:
    test_base = CIFAR100(root=DATA_ROOT, train=False, download=True, transform=transform_test_cifar)
test_loader = DataLoader(test_base, batch_size=BATCH_SIZE, shuffle=False)

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
    target_subset_id=args.target_subset_id,
    n_trials=N_TRIALS,
    seed=SEED,
    calculate_fisher_on=args.fisher_on,
    num_forgotten_clients=args.num_forgotten_clients,
    unlearned_model_name=args.unlearned_model_name,
    all_forgotten_loaders=all_forgotten_loaders,
    kill_output_neuron=args.kill_output_neuron,
    digit_metrics_only=args.digit_metrics_only,
    baseline_variant=args.baseline_variant,
    current_client_id=args.current_client_id if args.current_client_id is not None else args.target_subset_id,
)

print(f"Best α: {study.best_params['alpha']:.6f}")
print(f"Best λ: {study.best_params['lambda']:.6f}")
