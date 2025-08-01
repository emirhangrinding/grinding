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
        print("\n--- Fine-tuning subset head and encoder's final layers for MTL model ---")
        
        # Freeze all parameters initially
        for param in model.parameters():
            param.requires_grad = False
            
        # Unfreeze the subset_head
        for param in model.subset_head.parameters():
            param.requires_grad = True

        # Unfreeze the digit_head as well
        for param in model.digit_head.parameters():
            param.requires_grad = True
        
        # Unfreeze the final layer of the ResNet encoder
        for param in model.resnet.layer4.parameters():
            param.requires_grad = True
            
        # Unfreeze the feature projection layer if it exists
        if model.feature_proj is not None:
            for param in model.feature_proj.parameters():
                param.requires_grad = True
        
        # Collect all parameters that need gradients
        params_to_tune = [p for p in model.parameters() if p.requires_grad]

    else: # No-MTL
        print("\n--- Fine-tuning entire model for no-MTL model ---")
        params_to_tune = model.parameters()

    optimizer = optim.Adam(params_to_tune, lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        if is_mtl:
            # Set the entire model to eval mode first to freeze all BatchNorm layers
            model.eval()
            
            # Set the subset head to train mode
            model.subset_head.train()
            
            # Set the digit head to train mode
            model.digit_head.train()

            # Set the final encoder layer's BatchNorm layers to train mode
            for module in model.resnet.layer4.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.train()

            # Set the feature projection layer to train mode if it exists
            if model.feature_proj is not None:
                model.feature_proj.train()
        else:
            model.train()
        
        running_loss = 0.0
        for batch in retain_loader:
            optimizer.zero_grad()
            
            if is_mtl:
                inputs, labels, subset_labels = batch
                inputs, labels, subset_labels = inputs.to(device), labels.to(device), subset_labels.to(device)
                
                digit_logits, subset_logits, _ = model(inputs)
                
                loss_digit = criterion(digit_logits, labels)
                loss_subset = criterion(subset_logits, subset_labels)
                loss = loss_digit + loss_subset  # Joint loss
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