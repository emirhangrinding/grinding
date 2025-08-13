#!/usr/bin/env python3
import subprocess
import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST, CIFAR10

from utils import set_global_seed, SEED_DEFAULT
from data import generate_subdatasets, transform_mnist, transform_test_cifar
from models import StandardResNet
from evaluation import (
    calculate_overall_digit_classification_accuracy,
    calculate_digit_classification_accuracy,
    get_membership_attack_prob_train_only
)

def evaluate_baseline_model(model_path, dataset_name="CIFAR10", num_clients=10, 
                          target_client_id=0, batch_size=128, data_root="./data"):
    """
    Evaluate the baseline model with comprehensive metrics including MIA scores.
    """
    print(f"\n--- Evaluating baseline model: {model_path} ---")
    
    # Setup device and seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(SEED_DEFAULT)
    print(f"Using device: {device}")
    
    # Load model
    model = StandardResNet(dataset_name=dataset_name)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"✓ Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Generate data
    print("Setting up data loaders...")
    clients_data, _, full_dataset = generate_subdatasets(
        dataset_name=dataset_name,
        setting="non-iid",
        num_clients=num_clients,
        data_root=data_root
    )
    
    # Create target and other client data loaders
    target_key = f"client{target_client_id + 1}"
    target_indices = clients_data[target_key]
    
    # Combine all other clients' data
    other_indices = []
    for k, v in clients_data.items():
        if k != target_key:
            other_indices.extend(v)
    
    # Create datasets and loaders
    target_dataset = Subset(full_dataset, target_indices)
    other_dataset = Subset(full_dataset, other_indices)
    
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=False)
    other_loader = DataLoader(other_dataset, batch_size=batch_size, shuffle=False)
    
    # Test loader
    if dataset_name == "MNIST":
        test_base = MNIST(root=data_root, train=False, download=True, transform=transform_mnist)
    else:
        test_base = CIFAR10(root=data_root, train=False, download=True, transform=transform_test_cifar)
    test_loader = DataLoader(test_base, batch_size=batch_size, shuffle=False)
    
    # Calculate accuracies
    print("\n--- Calculating Accuracy Metrics ---")
    
    # Target client accuracy (the client we want to "forget")
    target_acc = calculate_overall_digit_classification_accuracy(model, target_loader, device)
    print(f"Accuracy on target client {target_client_id}: {target_acc:.4f}")
    
    # Other clients accuracy (clients we want to retain)
    other_acc = calculate_overall_digit_classification_accuracy(model, other_loader, device)
    print(f"Accuracy on other clients: {other_acc:.4f}")
    
    # Overall test accuracy
    test_acc = calculate_overall_digit_classification_accuracy(model, test_loader, device)
    print(f"Test set accuracy: {test_acc:.4f}")
    
    # Combined target vs other accuracy (for no-MTL, target_subset_id=None)
    combined_dataset = torch.utils.data.ConcatDataset([other_dataset, target_dataset])
    combined_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=False)
    
    # This will return 0.0 for target_acc and overall accuracy for other_acc for no-MTL models
    target_digit_acc, other_digit_acc = calculate_digit_classification_accuracy(
        model, combined_loader, device, target_subset_id=None
    )
    print(f"Combined evaluation - Target: {target_digit_acc:.4f}, Other: {other_digit_acc:.4f}")
    
    # Calculate MIA score
    print("\n--- Calculating Membership Inference Attack (MIA) Score ---")
    try:
        # For MIA, we use other_loader as "retain" and target_loader as "forget"
        mia_score = get_membership_attack_prob_train_only(other_loader, target_loader, model)
        print(f"MIA Attack Accuracy (target vs other clients): {mia_score:.2f}%")
        
    except Exception as e:
        print(f"✗ Error calculating MIA score: {e}")
        mia_score = -1
    
    print("\n--- Evaluation Summary ---")
    print(f"Target Client {target_client_id} Accuracy: {target_acc:.4f}")
    print(f"Other Clients Accuracy: {other_acc:.4f}")
    print(f"Test Set Accuracy: {test_acc:.4f}")
    print(f"MIA Attack Accuracy: {mia_score:.2f}%")
    print("--- End Evaluation ---\n")
    
    return {
        "target_acc": target_acc,
        "other_acc": other_acc,
        "test_acc": test_acc,
        "mia_score": mia_score
    }

def main():
    """
    Runs the complete workflow for training a baseline model and performing SSD unlearning.
    """
    print("Starting the full workflow...")

    # Define the model file path (should match what train_baseline_all_no_mtl.py saves)
    baseline_model_path = "/kaggle/input/no-mtl/pytorch/default/1/baseline_all_clients_model.h5"

    # Step 1: Train the baseline model on all clients (only if not already trained)
    if os.path.exists(baseline_model_path):
        print(f"\n--- Step 1: Baseline model already exists ---")
        print(f"Found existing model: {baseline_model_path}")
        print("Skipping training step...")
        
        # Evaluate the existing model
        metrics = evaluate_baseline_model(baseline_model_path)
        
    else:
        print("\n--- Step 1: Training baseline model on all clients ---")
        train_script = "train_baseline_all_no_mtl.py"
        train_command = f"python {train_script}"
        
        try:
            subprocess.run(train_command, shell=True, check=True)
            print(f"--- Successfully completed: {train_script} ---")
        except subprocess.CalledProcessError as e:
            print(f"--- Error running {train_script}: {e} ---")
            return

        # Verify the model was created
        if not os.path.exists(baseline_model_path):
            print(f"--- Error: Expected model file {baseline_model_path} was not created ---")
            return
            
        # Evaluate the newly trained model
        metrics = evaluate_baseline_model(baseline_model_path)

    # Step 2: Run SSD unlearning with Optuna tuning
    print("\n--- Step 2: Running SSD unlearning with Optuna tuning ---")
    import argparse
    parser = argparse.ArgumentParser(description="Run full no-MTL workflow")
    parser.add_argument("--fisher-on", type=str, choices=["subset", "digit"], default="subset", help="Task to compute Fisher Information on during SSD: 'subset' or 'digit'")
    args_cli, _ = parser.parse_known_args()

    tune_script = "run_ssd_tuning_no_mtl.py"
    tune_command = f"python {tune_script} --model-path {baseline_model_path} --fisher-on {args_cli.fisher_on}"
    
    try:
        subprocess.run(tune_command, shell=True, check=True)
        print(f"--- Successfully completed: {tune_script} ---")
    except subprocess.CalledProcessError as e:
        print(f"--- Error running {tune_script}: {e} ---")
        return

    # Step 3: Fine-tune the unlearned model
    print("\n--- Step 3: Fine-tuning the unlearned no-MTL model ---")
    unlearned_model_path = "unlearned_model_no_mtl.h5"  # This should be the output of run_ssd_tuning_no_mtl.py
    if not os.path.exists(unlearned_model_path):
        print(f"--- Warning: Unlearned model {unlearned_model_path} not found. Skipping fine-tuning. ---")
    else:
        finetune_script = "finetune.py"
        finetune_command = (
            f"python {finetune_script} "
            f"--model-path {unlearned_model_path} "
            f"--epochs 10 "
            f"--target-client-id 0" # As per the script
        )
        
        try:
            subprocess.run(finetune_command, shell=True, check=True)
            print(f"--- Successfully completed: {finetune_script} ---")
        except subprocess.CalledProcessError as e:
            print(f"--- Error running {finetune_script}: {e} ---")
            return

    print("\nFull workflow completed successfully!")

if __name__ == "__main__":
    main() 