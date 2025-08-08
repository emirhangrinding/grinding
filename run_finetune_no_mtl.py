#!/usr/bin/env python3
import os
import torch
from torch.utils.data import DataLoader, Subset

from utils import set_global_seed, SEED_DEFAULT
from data import generate_subdatasets
from models import StandardResNet
from finetune import finetune_model
from torchvision.datasets import CIFAR10
from data import transform_test_cifar
from evaluation import calculate_overall_digit_classification_accuracy, get_membership_attack_prob_train_only

def main():
    """
    Runs the fine-tuning workflow from a pre-unlearned model.
    """
    print("Starting the fine-tuning workflow for a pre-unlearned model...")

    # Define model paths
    unlearned_model_path = "/kaggle/input/unlearned/pytorch/default/1/unlearned_model_no_mtl.h5"
    
    # Setup device and seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(SEED_DEFAULT)
    print(f"Using device: {device}")

    # Load the pre-unlearned model
    model = StandardResNet(dataset_name="CIFAR10")
    try:
        model.load_state_dict(torch.load(unlearned_model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"✓ Successfully loaded pre-unlearned model from {unlearned_model_path}")
    except Exception as e:
        print(f"✗ Error loading pre-unlearned model: {e}")
        return

    # Generate data
    print("Setting up data loaders...")
    clients_data, _, full_dataset = generate_subdatasets(
        dataset_name="CIFAR10",
        setting="non-iid",
        num_clients=10,
        data_root="./data"
    )
    
    target_client_id = 0
    target_key = f"client{target_client_id + 1}"
    target_indices = clients_data[target_key]
    
    other_indices = []
    for k, v in clients_data.items():
        if k != target_key:
            other_indices.extend(v)
            
    retain_dataset = Subset(full_dataset, other_indices)
    forget_dataset = Subset(full_dataset, target_indices)
    
    retain_loader = DataLoader(retain_dataset, batch_size=128, shuffle=True)
    forget_loader = DataLoader(forget_dataset, batch_size=128, shuffle=True)
    
    test_base = CIFAR10(root="./data", train=False, download=True, transform=transform_test_cifar)
    test_loader = DataLoader(test_base, batch_size=128, shuffle=False)

    # Evaluate the pre-unlearned model
    print("\n--- Evaluating pre-unlearned model (no-MTL) ---")

    # Test set accuracy
    test_acc = calculate_overall_digit_classification_accuracy(model, test_loader, device)
    print(f"Test set accuracy: {test_acc:.4f}")

    # Digit accuracy on target subset
    target_digit_acc = calculate_overall_digit_classification_accuracy(model, forget_loader, device)
    print(f"Digit accuracy on target subset: {target_digit_acc:.4f}")

    # Digit accuracy on other subsets
    other_digit_acc = calculate_overall_digit_classification_accuracy(model, retain_loader, device)
    print(f"Digit accuracy on other subsets: {other_digit_acc:.4f}")

    # Train-only MIA Score
    mia_score = get_membership_attack_prob_train_only(
        retain_loader=retain_loader,
        forget_loader=forget_loader,
        model=model
    )
    print(f"Train-only MIA Score: {mia_score:.4f}")

    # Fine-tune the unlearned model
    print("\n--- Fine-tuning the unlearned no-MTL model ---")
    
    # Build mapping needed for detailed evaluation during fine-tuning (no-MTL uses it for uniform interface)
    forgotten_client_loaders = {target_client_id: forget_loader}

    finetuned_model = finetune_model(
        model=model,
        is_mtl=False,
        retain_loader=retain_loader,
        forget_loader=forget_loader,
        forgotten_client_loaders=forgotten_client_loaders,
        test_loader=test_loader,
        target_client_id=target_client_id,
        epochs=10,
        device=device,
    )
    
    # Save the fine-tuned model
    finetuned_model_path = "finetuned_model_no_mtl.h5"
    try:
        torch.save(finetuned_model.state_dict(), finetuned_model_path)
        print(f"✓ Successfully saved fine-tuned model to {finetuned_model_path}")
    except Exception as e:
        print(f"✗ Error saving fine-tuned model: {e}")

    print("\nFine-tuning workflow completed successfully!")

if __name__ == "__main__":
    main() 