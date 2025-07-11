#!/usr/bin/env python3
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision.datasets import MNIST, CIFAR10
import random
import copy

from utils import set_global_seed
from data import generate_subdatasets, MultiTaskDataset, transform_mnist, transform_test_cifar, create_subset_data_loaders
from models import MTL_Two_Heads_ResNet
from ssd import ssd_unlearn_subset
from evaluation import (
    calculate_digit_classification_accuracy,
    calculate_subset_identification_accuracy,
    calculate_overall_digit_classification_accuracy,
    get_membership_attack_prob_train_only
)

# --- Configuration ---
MODEL_PATH = "/kaggle/input/latest-medium/pytorch/default/1/model_medium.h5" 
DATASET_NAME = "CIFAR10"
TARGET_SUBSET_ID = 0
NUM_CLIENTS = 10
BATCH_SIZE = 128
DATA_ROOT = "./data"
HEAD_SIZE = "medium"
SEED = 42

# SSD Hyperparameters from user
# alpha is the exponent, lambda is the dampening constant.
# selection_weighting was 1.0 during tuning.
SSD_EXPONENT = 0.178419  # This was 'alpha' in tuning
SSD_DAMPENING = 2.610056 # This was 'lambda' in tuning
SSD_SELECTION_WEIGHTING = 1.0

# Finetuning Hyperparameters
FT_LR = 1e-4  # A common learning rate for fine-tuning
FT_EPOCHS = 1

def main():
    """Main function to run SSD and then finetune."""
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(SEED)
    print(f"Using device: {device}")

    # --- Data Loading (adapted from run_ssd_tuning.py) ---
    print("Loading and preparing data...")
    clients_data, _, full_dataset = generate_subdatasets(
        dataset_name=DATASET_NAME,
        setting="non-iid",
        num_clients=NUM_CLIENTS,
        data_root=DATA_ROOT
    )

    mtl_dataset = MultiTaskDataset(full_dataset, clients_data)
    
    # Create a reproducible train/val split
    dataset_size = len(mtl_dataset)
    indices = list(range(dataset_size))
    # Note: We don't shuffle here to ensure train/val split is consistent
    # random.seed(SEED)
    # random.shuffle(indices)
    train_size = int(0.8 * dataset_size)
    train_indices = indices[:train_size]

    train_dataset = Subset(mtl_dataset, train_indices)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    retain_loader, forget_loader = create_subset_data_loaders(train_loader, TARGET_SUBSET_ID)

    # --- Test Loader (adapted from run_ssd_tuning.py) ---
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

    # --- Load Model ---
    print("Loading pretrained model...")
    pretrained_model = MTL_Two_Heads_ResNet(dataset_name=DATASET_NAME, num_clients=NUM_CLIENTS, head_size=HEAD_SIZE)
    try:
        pretrained_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        print("Please ensure the model path is correct and the file is available.")
        return
    pretrained_model.to(device)
    pretrained_model.eval()

    # --- 1. Apply SSD Unlearning ---
    # The ssd_unlearn_subset function from ssd.py will be used here.
    # It returns the unlearned model and initial metrics.
    unlearned_model, ssd_metrics = ssd_unlearn_subset(
        pretrained_model=pretrained_model,
        retain_loader=retain_loader,
        forget_loader=forget_loader,
        target_subset_id=TARGET_SUBSET_ID,
        device=device,
        exponent=SSD_EXPONENT,
        dampening_constant=SSD_DAMPENING,
        selection_weighting=SSD_SELECTION_WEIGHTING,
        test_loader=test_loader,
        calculate_fisher_on="subset"
    )

    # --- 2. Fine-tune the subset head ---
    print("\n--- Starting fine-tuning of the subset head ---")
    
    # Freeze all parameters first
    for param in unlearned_model.parameters():
        param.requires_grad = False
    
    # Unfreeze only the subset_head parameters
    params_to_tune = []
    for name, param in unlearned_model.named_parameters():
        if name.startswith("subset_head."):
            param.requires_grad = True
            params_to_tune.append(param)
    
    if not params_to_tune:
        print("Warning: No parameters found for the subset head. Skipping fine-tuning.")
    else:
        print(f"Fine-tuning {len(params_to_tune)} parameter tensors in the subset head.")

        optimizer = optim.Adam(params_to_tune, lr=FT_LR)
        criterion = nn.CrossEntropyLoss()
        
        # Set the main model to eval mode to freeze batch norm stats,
        # but keep the head being tuned in train mode.
        unlearned_model.eval()
        unlearned_model.subset_head.train()
        
        for epoch in range(FT_EPOCHS):
            running_loss = 0.0
            for inputs, _, subset_labels in retain_loader:
                inputs, subset_labels = inputs.to(device), subset_labels.to(device)

                optimizer.zero_grad()
                _, subset_logits, _ = unlearned_model(inputs)
                loss = criterion(subset_logits, subset_labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(retain_loader.dataset)
            print(f"Fine-tuning Epoch {epoch+1}/{FT_EPOCHS}, Loss on Retain Set: {epoch_loss:.4f}")

    # --- 3. Final Evaluation ---
    print("\n--- Calculating final metrics after fine-tuning ---")
    unlearned_model.eval()

    # Create a temporary combined loader for evaluation on train data
    # to be consistent with deepclean.py evaluation style.
    combined_dataset = ConcatDataset([retain_loader.dataset, forget_loader.dataset])
    eval_loader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Calculate final metrics
    target_digit_acc, other_digit_acc = calculate_digit_classification_accuracy(unlearned_model, eval_loader, device, TARGET_SUBSET_ID)
    target_subset_acc, other_subset_acc = calculate_subset_identification_accuracy(unlearned_model, eval_loader, device, TARGET_SUBSET_ID)
    test_digit_acc = calculate_overall_digit_classification_accuracy(unlearned_model, test_loader, device)
    mia_score = get_membership_attack_prob_train_only(retain_loader, forget_loader, unlearned_model)

    print(f"Digit accuracy on target subset: {target_digit_acc:.4f}")
    print(f"Digit accuracy on other subsets: {other_digit_acc:.4f}")
    print(f"Subset ID accuracy on target subset: {target_subset_acc:.4f}")
    print(f"Subset ID accuracy on other subsets: {other_subset_acc:.4f}")
    print(f"[TEST] Overall digit accuracy: {test_digit_acc:.4f}")
    print(f"Train-only MIA Score on forget set: {mia_score:.4f}")
    print("\n--- Finetuning Script Finished ---")


if __name__ == "__main__":
    main() 