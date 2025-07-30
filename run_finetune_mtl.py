#!/usr/bin/env python3
import subprocess
import os

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

    # Step 1: Run SSD unlearning with optimized parameters
    print("\n--- Step 1: Running SSD unlearning for MTL model with optimized parameters ---")
    unlearn_script = "ssd.py"  # Assuming ssd.py handles the unlearning
    unlearn_command = (
        f"python {unlearn_script} "
        f"--model-path {baseline_model_path} "
        f"--is-mtl "
        f"--alpha {alpha} "
        f"--lambda_ {lambda_} "
        f"--unlearned-model-save-path {unlearned_model_path}"
    )
    
    try:
        subprocess.run(unlearn_command, shell=True, check=True)
        print(f"--- Successfully completed: {unlearn_script} ---")
    except subprocess.CalledProcessError as e:
        print(f"--- Error running {unlearn_script}: {e} ---")
        return

    # Step 2: Fine-tune the unlearned model
    print("\n--- Step 2: Fine-tuning the unlearned MTL model ---")
    if not os.path.exists(unlearned_model_path):
        print(f"--- Error: Unlearned model {unlearned_model_path} not found. ---")
        return
        
    finetune_script = "finetune.py"
    finetune_command = (
        f"python {finetune_script} "
        f"--model-path {unlearned_model_path} "
        f"--is-mtl "
        f"--epochs 10 "
        f"--target-client-id 0"  # As per the original script
    )
    
    try:
        subprocess.run(finetune_command, shell=True, check=True)
        print(f"--- Successfully completed: {finetune_script} ---")
    except subprocess.CalledProcessError as e:
        print(f"--- Error running {finetune_script}: {e} ---")

    print("\nMTL unlearning and fine-tuning workflow completed successfully!")

if __name__ == "__main__":
    main() 