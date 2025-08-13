#!/usr/bin/env python3
import argparse
import subprocess
import os
import torch
from torch.utils.data import DataLoader, Subset

from utils import set_global_seed, SEED_DEFAULT
from data import generate_subdatasets
from models import MTL_Two_Heads_ResNet
from evaluation import get_membership_attack_prob_train_only, calculate_overall_digit_classification_accuracy, calculate_digit_classification_accuracy

def evaluate_baseline_mtl_model(model_path, dataset_name="CIFAR10", num_clients=10, 
                                 target_client_id=0, batch_size=128, data_root="./data"):
    """
    Evaluate the baseline MTL model with comprehensive metrics including MIA scores.
    """
    print(f"\n--- Evaluating baseline MTL model: {model_path} ---")
    
    # Setup device and seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(SEED_DEFAULT)
    print(f"Using device: {device}")
    
    # Load model
    model = MTL_Two_Heads_ResNet(dataset_name=dataset_name, num_clients=num_clients, head_size="medium")
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"✓ Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return

    # Generate data
    print("Setting up data loaders for evaluation...")
    clients_data, _, full_dataset = generate_subdatasets(
        dataset_name=dataset_name,
        setting="non-iid", # Match training script
        num_clients=num_clients,
        data_root=data_root
    )
    
    # Create target and other client data loaders
    target_key = f"client{target_client_id + 1}"
    target_indices = clients_data[target_key]
    
    other_indices = []
    for i in range(num_clients):
        if i != target_client_id:
            other_indices.extend(clients_data[f"client{i + 1}"])

    target_dataset = Subset(full_dataset, target_indices)
    other_dataset = Subset(full_dataset, other_indices)
    
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=False)
    other_loader = DataLoader(other_dataset, batch_size=batch_size, shuffle=False)
    
    print("\n--- Calculating Accuracy Metrics ---")
    target_acc, _ = calculate_digit_classification_accuracy(model, target_loader, device, target_subset_id=target_client_id)
    print(f"Accuracy on target client {target_client_id}: {target_acc:.4f}")
    
    _, other_acc = calculate_digit_classification_accuracy(model, other_loader, device, target_subset_id=None) # Pass None for other clients
    print(f"Accuracy on other clients: {other_acc:.4f}")
    
    # Calculate MIA score
    print("\n--- Calculating Membership Inference Attack (MIA) Score ---")
    try:
        mia_score = get_membership_attack_prob_train_only(other_loader, target_loader, model)
        print(f"MIA Attack Accuracy (target vs other clients): {mia_score:.2f}%")
        
    except Exception as e:
        print(f"✗ Error calculating MIA score: {e}")
        mia_score = -1

    print("\n--- Evaluation Summary ---")
    print(f"Target Client {target_client_id} Accuracy: {target_acc:.4f}")
    print(f"Other Clients Accuracy: {other_acc:.4f}")
    print(f"MIA Attack Accuracy: {mia_score:.2f}%")
    print("--- End Evaluation ---\n")

    return {
        "target_acc": target_acc,
        "other_acc": other_acc,
        "mia_score": mia_score
    }

def main():
    """
    Runs the complete workflow for training a baseline MTL model and performing SSD unlearning.
    """
    parser = argparse.ArgumentParser(description="Run full MTL workflow")
    parser.add_argument("--disable-disentanglement", action="store_true", help="Disable disentanglement loss during baseline training")
    parser.add_argument("--fisher-on", type=str, choices=["subset", "digit"], default="subset", help="Task to compute Fisher Information on during SSD: 'subset' or 'digit'")
    parser.add_argument("--ce-only", action="store_true", help="Use CE-only baseline model for unlearning (uses Kaggle CE-only weights if present; trains with lambda_dis=0.0 otherwise)")
    args = parser.parse_args()

    print("Starting the full MTL workflow...")

    kaggle_default = "/kaggle/input/mtl/pytorch/default/1/baseline_mtl_all_clients.h5"
    kaggle_ce_only = "/kaggle/input/ce_only/pytorch/default/1/baseline_mtl_all_clients.h5"
    # Choose model path
    if args.ce_only:
        # Prefer Kaggle CE-only baseline if present; otherwise, use a local CE-only filename
        baseline_model_path = kaggle_ce_only if os.path.exists(kaggle_ce_only) else "baseline_mtl_all_clients_ce_only.h5"
    elif args.disable_disentanglement:
        baseline_model_path = "baseline_mtl_all_clients_no_dis.h5"
    else:
        baseline_model_path = kaggle_default if os.path.exists(kaggle_default) else "baseline_mtl_all_clients.h5"

    # Step 1: Train the baseline model on all clients
    if os.path.exists(baseline_model_path):
        print(f"\n--- Step 1: Baseline MTL model already exists ---")
        print(f"Found existing model: {baseline_model_path}")
        print("Skipping training step...")
        evaluate_baseline_mtl_model(baseline_model_path)
    else:
        print("\n--- Step 1: Training baseline MTL model on all clients ---")
        train_script = "train_baseline_all_mtl.py"
        train_command = f"python {train_script} --path {baseline_model_path}"
        # Train with lambda_dis=0.0 if CE-only is requested or disentanglement is disabled
        if args.ce_only or args.disable_disentanglement:
            train_command += " --lambda_dis 0.0"
        
        try:
            subprocess.run(train_command, shell=True, check=True)
            print(f"--- Successfully completed: {train_script} ---")
        except subprocess.CalledProcessError as e:
            print(f"--- Error running {train_script}: {e} ---")
            return

        if not os.path.exists(baseline_model_path):
            print(f"--- Error: Expected model file {baseline_model_path} was not created ---")
            return
        
        evaluate_baseline_mtl_model(baseline_model_path)

    # Step 2: Run SSD unlearning with Optuna tuning
    print("\n--- Step 2: Running SSD unlearning with Optuna tuning for MTL model ---")
    tune_script = "run_ssd_tuning.py"
    tune_command = f"python {tune_script} --model-path {baseline_model_path} --fisher-on {args.fisher_on}"
    
    try:
        subprocess.run(tune_command, shell=True, check=True)
        print(f"--- Successfully completed: {tune_script} ---")
    except subprocess.CalledProcessError as e:
        print(f"--- Error running {tune_script}: {e} ---")
        return

    # Step 3: Fine-tune the unlearned model
    print("\n--- Step 3: Fine-tuning the unlearned MTL model ---")
    unlearned_model_path = "unlearned_model_mtl.h5"  # This should be the output of run_ssd_tuning.py
    if not os.path.exists(unlearned_model_path):
        print(f"--- Warning: Unlearned model {unlearned_model_path} not found. Skipping fine-tuning. ---")
    else:
        finetune_script = "finetune.py"
        finetune_command = (
            f"python {finetune_script} "
            f"--model-path {unlearned_model_path} "
            f"--is-mtl "
            f"--epochs 10 "
            f"--target-client-id 0" # As per the script
        )
        
        try:
            subprocess.run(finetune_command, shell=True, check=True)
            print(f"--- Successfully completed: {finetune_script} ---")
        except subprocess.CalledProcessError as e:
            print(f"--- Error running {finetune_script}: {e} ---")
            return

    print("\nFull MTL workflow completed successfully!")

if __name__ == "__main__":
    main() 