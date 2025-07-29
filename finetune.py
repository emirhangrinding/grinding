#!/usr/bin/env python3
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, ConcatDataset
import argparse
import os

from utils import set_global_seed
from data import generate_subdatasets, MultiTaskDataset, create_subset_data_loaders
from models import MTL_Two_Heads_ResNet, StandardResNet
from evaluation import (
    calculate_digit_classification_accuracy,
    calculate_subset_identification_accuracy,
    calculate_overall_digit_classification_accuracy,
    get_membership_attack_prob_train_only
)

def finetune_model(
    model_path, 
    is_mtl, 
    dataset_name="CIFAR10", 
    num_clients=10, 
    target_client_id=0,
    batch_size=128, 
    data_root="./data", 
    epochs=10, 
    lr=1e-4, 
    head_size="medium",
    seed=42
):
    """
    Fine-tunes a model after unlearning.
    - For MTL models, it fine-tunes the subset head on the retain data.
    - For no-MTL models, it fine-tunes the entire model on the retain data.
    """
    print(f"\n--- Starting fine-tuning for {'MTL' if is_mtl else 'No-MTL'} model: {model_path} ---")

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(seed)
    print(f"Using device: {device}")

    # --- Data Loading ---
    print("Loading and preparing data...")
    clients_data, _, full_dataset = generate_subdatasets(
        dataset_name=dataset_name,
        setting="non-iid",
        num_clients=num_clients,
        data_root=data_root
    )

    if is_mtl:
        mtl_dataset = MultiTaskDataset(full_dataset, clients_data)
        train_loader = DataLoader(mtl_dataset, batch_size=batch_size, shuffle=True)
        retain_loader, forget_loader = create_subset_data_loaders(train_loader, target_client_id)
    else:
        # For no-MTL, we just need the retain and forget loaders.
        target_key = f"client{target_client_id + 1}"
        target_indices = clients_data[target_key]
        
        other_indices = []
        for i in range(num_clients):
            if i != target_client_id:
                other_indices.extend(clients_data[f"client{i + 1}"])

        retain_dataset = Subset(full_dataset, other_indices)
        forget_dataset = Subset(full_dataset, target_indices)
        retain_loader = DataLoader(retain_dataset, batch_size=batch_size, shuffle=True)
        forget_loader = DataLoader(forget_dataset, batch_size=batch_size, shuffle=False)
        
        combined_dataset = ConcatDataset([retain_dataset, forget_dataset])
        eval_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=False)


    # --- Load Model ---
    print(f"Loading unlearned model from {model_path}...")
    if is_mtl:
        model = MTL_Two_Heads_ResNet(dataset_name=dataset_name, num_clients=num_clients, head_size=head_size)
    else:
        model = StandardResNet(dataset_name=dataset_name)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {model_path}")
        return
    model.to(device)

    # --- Fine-tuning ---
    if is_mtl:
        print("\n--- Fine-tuning subset head for MTL model ---")
        # Freeze all parameters first
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze only the subset_head parameters
        params_to_tune = []
        for name, param in model.named_parameters():
            if name.startswith("subset_head."):
                param.requires_grad = True
                params_to_tune.append(param)
        
        if not params_to_tune:
            print("Warning: No parameters found for subset head. Skipping fine-tuning.")
            return

        optimizer = optim.Adam(params_to_tune, lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        model.eval()
        model.subset_head.train()

        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, _, subset_labels in retain_loader:
                inputs, subset_labels = inputs.to(device), subset_labels.to(device)
                optimizer.zero_grad()
                _, subset_logits, _ = model(inputs)
                loss = criterion(subset_logits, subset_labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            
            epoch_loss = running_loss / len(retain_loader.dataset)
            print(f"Fine-tuning Epoch {epoch+1}/{epochs}, Loss on Retain Set: {epoch_loss:.4f}")

    else: # No-MTL
        print("\n--- Fine-tuning entire model for no-MTL model ---")
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in retain_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs, _ = model(inputs) # No subset logits for standard model
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            
            epoch_loss = running_loss / len(retain_loader.dataset)
            print(f"Fine-tuning Epoch {epoch+1}/{epochs}, Loss on Retain Set: {epoch_loss:.4f}")

    # --- Evaluation after fine-tuning ---
    print("\n--- Calculating metrics after fine-tuning ---")
    model.eval()

    if is_mtl:
        target_digit_acc, other_digit_acc = calculate_digit_classification_accuracy(model, train_loader, device, target_client_id)
        target_subset_acc, other_subset_acc = calculate_subset_identification_accuracy(model, train_loader, device, target_client_id)
        mia_score = get_membership_attack_prob_train_only(retain_loader, forget_loader, model)

        print(f"Digit accuracy on target subset: {target_digit_acc:.4f}")
        print(f"Digit accuracy on other subsets: {other_digit_acc:.4f}")
        print(f"Subset ID accuracy on target subset: {target_subset_acc:.4f}")
        print(f"Subset ID accuracy on other subsets: {other_subset_acc:.4f}")
        print(f"Train-only MIA Score: {mia_score:.4f}")
    else:
        target_digit_acc = calculate_overall_digit_classification_accuracy(model, forget_loader, device)
        other_digit_acc = calculate_overall_digit_classification_accuracy(model, retain_loader, device)
        mia_score = get_membership_attack_prob_train_only(retain_loader, forget_loader, model)

        print(f"Digit accuracy on forgotten data: {target_digit_acc:.4f}")
        print(f"Digit accuracy on retained data: {other_digit_acc:.4f}")
        print(f"Train-only MIA Score: {mia_score:.4f}")


    # Save the fine-tuned model
    finetuned_model_path = model_path.replace(".h5", "_finetuned.h5")
    torch.save(model.state_dict(), finetuned_model_path)
    print(f"✓ Fine-tuned model saved to {finetuned_model_path}")
    print("\n--- Fine-tuning Script Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a model after unlearning.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the unlearned model file.")
    parser.add_argument("--is-mtl", action="store_true", help="Flag to indicate if the model is a Multi-Task Learning model.")
    parser.add_argument("--dataset", type=str, default="CIFAR10", choices=["CIFAR10", "MNIST"], help="Dataset to use.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of fine-tuning epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for fine-tuning.")
    parser.add_argument("--target-client-id", type=int, default=0, help="The client ID that was forgotten.")
    
    args = parser.parse_args()

    finetune_model(
        model_path=args.model_path,
        is_mtl=args.is_mtl,
        dataset_name=args.dataset,
        epochs=args.epochs,
        lr=args.lr,
        target_client_id=args.target_client_id
    ) 