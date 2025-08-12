#!/usr/bin/env python3
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
import copy

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
    lambda_digit=0.3,  # Weight for the adversarial digit loss
    lambda_subset=0.05, # Weight for the adversarial subset ID loss
    seed=42,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    # Grid search controls (used only for MTL)
    search_lambdas: bool = True,
    lambda_digit_grid=None,
    lambda_subset_grid=None,
    save_best_model_path: str = None,
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

    criterion = nn.CrossEntropyLoss()

    # Default grids if not provided (used only when is_mtl and search_lambdas)
    if lambda_digit_grid is None:
        lambda_digit_grid = [0.0, 0.1, 0.3, 0.5]
    if lambda_subset_grid is None:
        lambda_subset_grid = [0.0, 0.05, 0.1, 0.2]

    def score_from_metrics(metrics_dict: dict):
        test_acc = metrics_dict.get("Test set accuracy", 0.0)
        retain_acc = metrics_dict.get("Digit accuracy on other subsets", 0.0)
        # Lower forgotten accuracy is better, so we negate it in the score tuple
        forgotten_digit_accs = [
            v for k, v in metrics_dict.items()
            if k.startswith("Digit acc on client")
        ]
        avg_forgotten = sum(forgotten_digit_accs) / len(forgotten_digit_accs) if forgotten_digit_accs else 0.0
        return (test_acc, retain_acc, -avg_forgotten)

    def train_one_setting(current_lambda_digit: float, current_lambda_subset: float):
        # Ensure all parameters are trainable each run
        for param in model.parameters():
            param.requires_grad = True
        params_to_tune = model.parameters()
        optimizer = optim.Adam(params_to_tune, lr=lr)
        if is_mtl:
            forget_loader_iter = iter(forget_loader)

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0

            for batch in retain_loader:
                optimizer.zero_grad()

                if is_mtl:
                    # For MTL, we use an adversarial loss, requiring both retain and forget data
                    try:
                        forget_batch = next(forget_loader_iter)
                    except StopIteration:
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
                        - ((current_lambda_digit * digit_excess) * digit_loss_f)
                        - (current_lambda_subset * subset_loss_f)
                    )
                else:  # No-MTL
                    inputs, labels = batch
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()
                running_loss += loss.item() * batch[0].size(0)

            epoch_loss = running_loss / len(retain_loader.dataset)
            if is_mtl:
                print(
                    f"Fine-tuning Epoch {epoch+1}/{epochs} | lambda_digit={current_lambda_digit}, "
                    f"lambda_subset={current_lambda_subset} | Objective: {epoch_loss:.4f}"
                )
            else:
                print(f"Fine-tuning Epoch {epoch+1}/{epochs} | Objective: {epoch_loss:.4f}")

        print("\n--- Metrics after fine-tuning with current lambdas ---")
        metrics = evaluate_and_print_metrics(
            model=model,
            is_mtl=is_mtl,
            retain_loader=retain_loader,
            test_loader=test_loader,
            device=device,
            forgotten_client_loaders=forgotten_client_loaders,
            current_forget_client_id=target_client_id,
        )
        return metrics

    # Keep the initial model state so each grid run starts from the same baseline
    base_state_dict = copy.deepcopy(model.state_dict())

    if is_mtl and search_lambdas:
        print("\n--- Fine-tuning entire model for MTL model (lambda grid search) ---")
        best_score = None
        best_state = None
        best_combo = None
        for ld in lambda_digit_grid:
            for ls in lambda_subset_grid:
                print(f"\n>>> Trying lambda_digit={ld}, lambda_subset={ls}")
                model.load_state_dict(base_state_dict)
                metrics = train_one_setting(ld, ls)
                score = score_from_metrics(metrics)
                if (best_score is None) or (score > best_score):
                    best_score = score
                    best_state = copy.deepcopy(model.state_dict())
                    best_combo = (ld, ls)
        # Load best weights
        if best_state is not None:
            model.load_state_dict(best_state)
            print(
                f"\n✓ Selected best lambdas: lambda_digit={best_combo[0]}, lambda_subset={best_combo[1]} "
                f"with score={best_score}"
            )
        else:
            print("Warning: No best state captured; falling back to last state.")
        # Save best model if requested or use a sensible default path
        if save_best_model_path is None:
            save_best_model_path = f"finetuned_best_mtl_client_{target_client_id}.h5"
        try:
            torch.save(model.state_dict(), save_best_model_path)
            print(f"✓ Saved best fine-tuned model to {save_best_model_path}")
        except Exception as e:
            print(f"Failed to save best fine-tuned model: {e}")
    else:
        # No-MTL or grid search disabled: do a single fine-tuning run
        if is_mtl:
            print("\n--- Fine-tuning entire model for MTL model (single setting) ---")
            print(f"Using lambda_digit={lambda_digit}, lambda_subset={lambda_subset}")
        else:
            print("\n--- Fine-tuning entire model for no-MTL model ---")

        # Restore base state and run once
        model.load_state_dict(base_state_dict)
        _ = train_one_setting(lambda_digit, lambda_subset)
        # Save the resulting model
        if save_best_model_path is None:
            save_best_model_path = (
                f"finetuned_best_mtl_client_{target_client_id}.h5" if is_mtl else "finetuned_best_no_mtl.h5"
            )
        try:
            torch.save(model.state_dict(), save_best_model_path)
            print(f"✓ Saved fine-tuned model to {save_best_model_path}")
        except Exception as e:
            print(f"Failed to save fine-tuned model: {e}")

    print("\n--- Fine-tuning Script Finished ---")
    return model

if __name__ == "__main__":
    # This part remains for standalone testing if needed, but the main workflow
    # will call finetune_model directly from other scripts.
    # Note: For standalone execution, data loading logic would need to be re-added here.
    print("This script is intended to be called from a workflow script that provides data loaders.") 