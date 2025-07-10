#!/usr/bin/env python3
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST, CIFAR10

from utils import set_global_seed, SEED_DEFAULT
from data import generate_subdatasets, MultiTaskDataset, transform_mnist, transform_test_cifar, create_subset_data_loaders
from models import MTL_Two_Heads_ResNet
from tuning import optimise_ssd_hyperparams
from ssd import ssd_unlearn_subset

# Configuration - EDIT THESE VALUES
MODEL_PATH = "/kaggle/input/latest-medium/pytorch/default/1/model_medium.h5" 
DATASET_NAME = "CIFAR10"
TARGET_SUBSET_ID = 0
NUM_CLIENTS = 10
BATCH_SIZE = 128
N_TRIALS = 100
DATA_ROOT = "./data"
HEAD_SIZE = "medium"
SEED = 42
CALCULATE_FISHER_ON = "subset"

# Finetuning configuration
FINETUNE_EPOCHS = 10
FINETUNE_LR = 1e-3

def finetune_subset_identification(model, retain_loader, device, epochs=10, lr=1e-3):
    """
    Finetune the model for subset identification task using Adam optimizer.
    
    Args:
        model: The model to finetune
        retain_loader: DataLoader for retain data
        device: Device to run on
        epochs: Number of epochs for finetuning
        lr: Learning rate for Adam optimizer
    
    Returns:
        The finetuned model
    """
    print(f"\n--- Starting {epochs} epochs of finetuning for subset identification ---")
    
    # Set up optimizer for all model parameters
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_idx, (inputs, digit_labels, subset_labels) in enumerate(retain_loader):
            inputs = inputs.to(device)
            subset_labels = subset_labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass - get both outputs but only use subset classification
            digit_logits, subset_logits, _ = model(inputs)
            
            # Loss based on subset identification only
            loss = criterion(subset_logits, subset_labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track metrics
            running_loss += loss.item()
            _, predicted = torch.max(subset_logits, 1)
            correct_predictions += (predicted == subset_labels).sum().item()
            total_samples += subset_labels.size(0)
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(retain_loader)
        epoch_acc = correct_predictions / total_samples
        
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f}, Subset Accuracy: {epoch_acc:.4f}")
    
    print("--- Finetuning completed ---")
    return model

# Setup
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
random.shuffle(indices)
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
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Run optimization
print("Running SSD hyperparameter optimization...")
study = optimise_ssd_hyperparams(
    pretrained_model=model,
    retain_loader=retain_loader,
    forget_loader=forget_loader,
    test_loader=test_loader,
    device=device,
    target_subset_id=TARGET_SUBSET_ID,
    n_trials=N_TRIALS,
    seed=SEED,
    calculate_fisher_on=CALCULATE_FISHER_ON 
)

print(f"\nBest α: {study.best_params['alpha']:.6f}")
print(f"Best λ: {study.best_params['lambda']:.6f}")

# Apply SSD with best parameters to get the final unlearned model
print("\nApplying SSD with best parameters...")
best_alpha = study.best_params['alpha']
best_lambda = study.best_params['lambda']

unlearned_model, ssd_metrics = ssd_unlearn_subset(
    pretrained_model=model,
    retain_loader=retain_loader,
    forget_loader=forget_loader,
    target_subset_id=TARGET_SUBSET_ID,
    device=device,
    lower_bound=1.0,
    exponent=best_alpha,
    dampening_constant=best_lambda,
    selection_weighting=1.0,
    test_loader=test_loader,
    calculate_fisher_on=CALCULATE_FISHER_ON
)

print("\nSSD Results:")
print(f"Target Digit Accuracy: {ssd_metrics['target_digit_acc']:.4f}")
print(f"Other Digit Accuracy: {ssd_metrics['other_digit_acc']:.4f}")
print(f"Target Subset Accuracy: {ssd_metrics['target_subset_acc']:.4f}")
print(f"Other Subset Accuracy: {ssd_metrics['other_subset_acc']:.4f}")
print(f"Test Digit Accuracy: {ssd_metrics['test_digit_acc']:.4f}")
print(f"MIA Score: {ssd_metrics['mia_score']:.2f}%")

# Finetune the unlearned model for subset identification
finetuned_model = finetune_subset_identification(
    model=unlearned_model,
    retain_loader=retain_loader,
    device=device,
    epochs=FINETUNE_EPOCHS,
    lr=FINETUNE_LR
)

print("\nFinetuning completed! The model has been trained for 10 additional epochs focusing on subset identification.")