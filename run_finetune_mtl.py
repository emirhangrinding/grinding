#!/usr/bin/env python3
import os
import torch
from torch.utils.data import DataLoader, Subset

from utils import set_global_seed, SEED_DEFAULT
from data import generate_subdatasets, MultiTaskDataset, transform_test_cifar
from models import MTL_Two_Heads_ResNet
from ssd import ssd_unlearn_subset
from finetune import finetune_model
from torchvision.datasets import CIFAR10

def main():
    """
    Runs the unlearning and fine-tuning workflow for an MTL model with optimized hyperparameters.
    """
    print("Starting the MTL unlearning and fine-tuning workflow with optimized parameters...")

    # Optimized hyperparameters for MTL
    alpha = 2.80069532231228
    lambda_ = 2.6653773171952015

    # Define model paths
    baseline_model_path = "/kaggle/input/mtl/pytorch/default/1/baseline_mtl_all_clients.h5"
    unlearned_model_path = "unlearned_model_mtl_optimal.h5"

    # Setup device and seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(SEED_DEFAULT)
    print(f"Using device: {device}")

    # Load baseline model
    num_clients = 10
    model = MTL_Two_Heads_ResNet(dataset_name="CIFAR10", num_clients=num_clients, head_size="medium")
    try:
        model.load_state_dict(torch.load(baseline_model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"✓ Successfully loaded baseline model from {baseline_model_path}")
    except Exception as e:
        print(f"✗ Error loading baseline model: {e}")
        return

    # Generate data
    print("Setting up data loaders...")
    clients_data, _, full_dataset = generate_subdatasets(
        dataset_name="CIFAR10",
        setting="non-iid",
        num_clients=num_clients,
        data_root="./data"
    )
    
    target_client_id = 0
    target_key = f"client{target_client_id + 1}"
    target_indices = clients_data[target_key]
    
    other_indices = []
    for i in range(num_clients):
        if i != target_client_id:
            other_indices.extend(clients_data[f"client{i + 1}"])

    multitask_dataset = MultiTaskDataset(full_dataset, clients_data)
    retain_dataset = Subset(multitask_dataset, other_indices)
    forget_dataset = Subset(multitask_dataset, target_indices)
    
    retain_loader = DataLoader(retain_dataset, batch_size=128, shuffle=True)
    forget_loader = DataLoader(forget_dataset, batch_size=128, shuffle=True)
    
    test_base = CIFAR10(root="./data", train=False, download=True, transform=transform_test_cifar)
    test_loader = DataLoader(test_base, batch_size=128, shuffle=False)

    # Step 1: Run SSD unlearning with optimized parameters
    print("\n--- Step 1: Running SSD unlearning for MTL model with optimized parameters ---")
    
    unlearned_model, _ = ssd_unlearn_subset(
        pretrained_model=model,
        retain_loader=retain_loader,
        forget_loader=forget_loader,
        target_subset_id=target_client_id,  # For MTL
        device=device,
        selection_weighting=alpha,
        dampening_constant=lambda_,
        calculate_fisher_on="subset"
    )

    # Step 2: Fine-tune the unlearned model
    print("\n--- Step 2: Fine-tuning the unlearned MTL model ---")
    
    finetuned_model = finetune_model(
        model=unlearned_model,
        is_mtl=True,
        retain_loader=retain_loader,
        forget_loader=forget_loader,
        test_loader=test_loader,
        target_client_id=target_client_id,
        epochs=10,
        device=device,
    )

    # Save the fine-tuned model
    finetuned_model_path = unlearned_model_path.replace(".h5", "_finetuned.h5")
    try:
        torch.save(finetuned_model.state_dict(), finetuned_model_path)
        print(f"✓ Successfully saved fine-tuned model to {finetuned_model_path}")
    except Exception as e:
        print(f"✗ Error saving fine-tuned model: {e}")
        
    print("\nMTL unlearning and fine-tuning workflow completed successfully!")

if __name__ == "__main__":
    main() 