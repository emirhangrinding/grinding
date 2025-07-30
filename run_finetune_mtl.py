#!/usr/bin/env python3
import subprocess
import os
import torch
from torch.utils.data import DataLoader, Subset

from utils import set_global_seed, SEED_DEFAULT
from data import generate_subdatasets, MultiTaskDataset
from models import MTL_Two_Heads_ResNet
from ssd import ssd_unlearn_subset

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

    retain_dataset = MultiTaskDataset(Subset(full_dataset, other_indices), subset_id=None, num_subsets=num_clients)
    forget_dataset = MultiTaskDataset(Subset(full_dataset, target_indices), subset_id=target_client_id, num_subsets=num_clients)
    
    retain_loader = DataLoader(retain_dataset, batch_size=128, shuffle=True)
    forget_loader = DataLoader(forget_dataset, batch_size=128, shuffle=True)

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

    # Save the unlearned model
    try:
        torch.save(unlearned_model.state_dict(), unlearned_model_path)
        print(f"✓ Successfully saved unlearned model to {unlearned_model_path}")
    except Exception as e:
        print(f"✗ Error saving unlearned model: {e}")
        return
        
    # Step 2: Fine-tune the unlearned model
    print("\n--- Step 2: Fine-tuning the unlearned MTL model ---")
    
    finetune_script = "finetune.py"
    finetune_command = (
        f"python {finetune_script} "
        f"--model-path {unlearned_model_path} "
        f"--is-mtl "
        f"--epochs 10 "
        f"--target-client-id {target_client_id}"
    )
    
    try:
        subprocess.run(finetune_command, shell=True, check=True)
        print(f"--- Successfully completed: {finetune_script} ---")
    except subprocess.CalledProcessError as e:
        print(f"--- Error running {finetune_script}: {e} ---")

    print("\nMTL unlearning and fine-tuning workflow completed successfully!")

if __name__ == "__main__":
    main() 