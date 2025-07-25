#!/usr/bin/env python3
import subprocess
import os

def main():
    """
    Runs the complete workflow for training a baseline MTL model and performing SSD unlearning.
    """
    print("Starting the full MTL workflow...")

    baseline_model_path = "baseline_mtl_all_clients.h5"

    # Step 1: Train the baseline model on all clients
    if os.path.exists(baseline_model_path):
        print(f"\n--- Step 1: Baseline MTL model already exists ---")
        print(f"Found existing model: {baseline_model_path}")
        print("Skipping training step...")
    else:
        print("\n--- Step 1: Training baseline MTL model on all clients ---")
        train_script = "train_baseline_all_mtl.py"
        train_command = f"python {train_script}"
        
        try:
            subprocess.run(train_command, shell=True, check=True)
            print(f"--- Successfully completed: {train_script} ---")
        except subprocess.CalledProcessError as e:
            print(f"--- Error running {train_script}: {e} ---")
            return

        if not os.path.exists(baseline_model_path):
            print(f"--- Error: Expected model file {baseline_model_path} was not created ---")
            return

    # Step 2: Run SSD unlearning with Optuna tuning
    print("\n--- Step 2: Running SSD unlearning with Optuna tuning for MTL model ---")
    tune_script = "run_ssd_tuning.py"
    tune_command = f"python {tune_script}"
    
    try:
        subprocess.run(tune_command, shell=True, check=True)
        print(f"--- Successfully completed: {tune_script} ---")
    except subprocess.CalledProcessError as e:
        print(f"--- Error running {tune_script}: {e} ---")
        return

    print("\nFull MTL workflow completed successfully!")

if __name__ == "__main__":
    main() 