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
    evaluate_and_print_metrics,
)
from utils import set_global_seed

def finetune_model(
    model,
    is_mtl,
    retain_loader,
    forget_loader,
    forgotten_client_loaders,
    test_loader,
    target_client_id,
    epochs=1,
    lr=1e-4,
    lambda_digit=0.1,  # Weight for the adversarial digit loss
    lambda_subset=0.1, # Weight for the adversarial subset ID loss
    seed=42,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    """
    Fine-tunes a model after unlearning, using retain data from the remaining clients
    to improve performance, while adversarially penalizing performance on the currently
    forgotten client's data.
    """
    set_global_seed(seed)
    model.to(device)

    # Establish a baseline digit accuracy for the current forgotten client (client_id)
    # We will only penalize the adversarial digit loss when the current accuracy exceeds this baseline.
    baseline_digit_acc_forget = calculate_overall_digit_classification_accuracy(
        model, forget_loader, device
    ) if is_mtl else 0.0
    if is_mtl:
        print(f"Baseline digit acc on forgotten client {target_client_id}: {baseline_digit_acc_forget:.4f}")

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

    # Initialize an iterator for the forget_loader for use in the adversarial loss
    if is_mtl:
        forget_loader_iter = iter(forget_loader)

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
                # For MTL, we use an adversarial loss, requiring both retain and forget data
                try:
                    forget_batch = next(forget_loader_iter)
                except StopIteration:
                    # Replenish the iterator when it's exhausted
                    forget_loader_iter = iter(forget_loader)
                    forget_batch = next(forget_loader_iter)

                # --- Retain Set Loss (Encourage Learning) ---
                inputs_r, labels_r, subset_labels_r = batch
                inputs_r, labels_r, subset_labels_r = inputs_r.to(device), labels_r.to(device), subset_labels_r.to(device)
                digit_logits_r, subset_logits_r, _ = model(inputs_r)
                loss_digit_r = criterion(digit_logits_r, labels_r)
                loss_subset_r = criterion(subset_logits_r, subset_labels_r)
                retain_loss = loss_digit_r + loss_subset_r

                # --- Forget Set Loss (Adversarial: Encourage Forgetting) ---
                inputs_f, labels_f, subset_labels_f = forget_batch
                inputs_f, labels_f, subset_labels_f = inputs_f.to(device), labels_f.to(device), subset_labels_f.to(device)
                
                digit_logits_f, subset_logits_f, _ = model(inputs_f)

                # Compute batch accuracy on the forgotten client to modulate the adversarial digit penalty
                with torch.no_grad():
                    _, digit_preds_f = torch.max(digit_logits_f, 1)
                    batch_acc_f = (digit_preds_f == labels_f).float().mean().item()
                    # Scale digit penalty by the amount current acc exceeds baseline; 0 if below baseline
                    digit_excess = max(0.0, batch_acc_f - baseline_digit_acc_forget)
                
                digit_loss_f = criterion(digit_logits_f, labels_f)
                subset_loss_f = criterion(subset_logits_f, subset_labels_f)

                # --- Combined Loss ---
                loss = (
                    retain_loss
                    - ((lambda_digit * digit_excess) * digit_loss_f)
                    - (lambda_subset * subset_loss_f)
                )
                
            else: # No-MTL
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch[0].size(0)

        epoch_loss = running_loss / len(retain_loader.dataset)
        print(f"Fine-tuning Epoch {epoch+1}/{epochs}, Combined objective (retain - lambda_d*forget_digit - lambda_s*forget_subset): {epoch_loss:.4f}")

        print(f"\n--- Metrics after fine-tuning epoch {epoch+1}/{epochs} ---")
        # Use the shared evaluation to report per-client metrics (forgotten clients individually + others)
        evaluate_and_print_metrics(
            model=model,
            is_mtl=is_mtl,
            retain_loader=retain_loader,
            test_loader=test_loader,
            device=device,
            forgotten_client_loaders=forgotten_client_loaders,
            current_forget_client_id=target_client_id,
        )
        
    print("\n--- Fine-tuning Script Finished ---")
    return model

if __name__ == "__main__":
    # This part remains for standalone testing if needed, but the main workflow
    # will call finetune_model directly from other scripts.
    # Note: For standalone execution, data loading logic would need to be re-added here.
    print("This script is intended to be called from a workflow script that provides data loaders.") 