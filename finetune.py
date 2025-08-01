#!/usr/bin/env python3
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os

from models import MTL_Two_Heads_ResNet, StandardResNet
from evaluation import (
    calculate_digit_classification_accuracy,
    calculate_subset_identification_accuracy,
    calculate_overall_digit_classification_accuracy,
    get_membership_attack_prob_train_only,
)
from utils import set_global_seed

def evaluate_and_print_metrics(
    model, 
    is_mtl, 
    retain_loader, 
    forget_loader, 
    test_loader, 
    device, 
    target_client_id
):
    """Helper function to evaluate model and print metrics."""
    model.eval()
    
    if is_mtl:
        # For MTL, we need a combined loader for evaluation that preserves subset labels
        # This approach is simplified; a more robust solution might be needed depending on the dataset structure
        combined_loader = DataLoader(
            torch.utils.data.ConcatDataset([retain_loader.dataset, forget_loader.dataset]),
            batch_size=retain_loader.batch_size,
            shuffle=False
        )
        target_digit_acc, other_digit_acc = calculate_digit_classification_accuracy(model, combined_loader, device, target_client_id)
        target_subset_acc, other_subset_acc = calculate_subset_identification_accuracy(model, combined_loader, device, target_client_id)
    else: # No-MTL
        target_digit_acc = calculate_overall_digit_classification_accuracy(model, forget_loader, device)
        other_digit_acc = calculate_overall_digit_classification_accuracy(model, retain_loader, device)
        target_subset_acc, other_subset_acc = 0.0, 0.0 # Not applicable

    mia_score = get_membership_attack_prob_train_only(retain_loader, forget_loader, model)
    test_acc = calculate_overall_digit_classification_accuracy(model, test_loader, device) if test_loader else -1.0

    print(f"Digit accuracy on target subset: {target_digit_acc:.4f}")
    print(f"Digit accuracy on other subsets: {other_digit_acc:.4f}")
    if is_mtl:
        print(f"Subset ID accuracy on target subset: {target_subset_acc:.4f}")
        print(f"Subset ID accuracy on other subsets: {other_subset_acc:.4f}")
    print(f"Test set accuracy: {test_acc:.4f}")
    print(f"Train-only MIA Score: {mia_score:.4f}")

def finetune_model(
    model,
    is_mtl,
    retain_loader,
    forget_loader,
    test_loader,
    target_client_id,
    epochs=10,
    lr=1e-4,
    seed=42,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    """
    Fine-tunes a model after unlearning, now accepting data loaders directly.
    """
    set_global_seed(seed)
    model.to(device)

    if is_mtl:
        print("\n--- Fine-tuning subset head for MTL model ---")
        params_to_tune = [param for name, param in model.named_parameters() if name.startswith("subset_head.")]
        if not params_to_tune:
            print("Warning: No parameters found for subset head. Skipping fine-tuning.")
            return model
    else: # No-MTL
        print("\n--- Fine-tuning entire model for no-MTL model ---")
        params_to_tune = model.parameters()

    optimizer = optim.Adam(params_to_tune, lr=lr)
    criterion = nn.CrossEntropyLoss()

    # If MTL, set requires_grad appropriately once before the loop
    if is_mtl:
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if name.startswith("subset_head."):
                param.requires_grad = True

    for epoch in range(epochs):
        if is_mtl:
            model.eval()
            model.subset_head.train()
        else:
            model.train()
        
        running_loss = 0.0
        for batch in retain_loader:
            optimizer.zero_grad()
            
            if is_mtl:
                inputs, _, subset_labels = batch
                inputs, subset_labels = inputs.to(device), subset_labels.to(device)
                _, subset_logits, _ = model(inputs)
                loss = criterion(subset_logits, subset_labels)
            else: # No-MTL
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(retain_loader.dataset)
        print(f"Fine-tuning Epoch {epoch+1}/{epochs}, Loss on Retain Set: {epoch_loss:.4f}")

        print(f"\n--- Metrics after fine-tuning epoch {epoch+1}/{epochs} ---")
        evaluate_and_print_metrics(model, is_mtl, retain_loader, forget_loader, test_loader, device, target_client_id)
        
    print("\n--- Fine-tuning Script Finished ---")
    return model

if __name__ == "__main__":
    # This part remains for standalone testing if needed, but the main workflow
    # will call finetune_model directly from other scripts.
    # Note: For standalone execution, data loading logic would need to be re-added here.
    print("This script is intended to be called from a workflow script that provides data loaders.") 