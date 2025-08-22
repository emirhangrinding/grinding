#!/usr/bin/env python3
import subprocess
import os
import torch
from torch.utils.data import DataLoader, Subset

from utils import set_global_seed, SEED_DEFAULT
from data import generate_subdatasets
from models import StandardResNet
from finetune import finetune_model
from evaluation import evaluate_and_print_metrics
from torchvision.datasets import MNIST, CIFAR10
from data import transform_mnist, transform_test_cifar
from train_baseline_all_no_mtl import learn_baseline_all_clients as train_no_mtl_baseline_all_clients

# --- Configuration ---
IS_MTL = False
DATASET_NAME = "CIFAR10"
NUM_CLIENTS = 10
BATCH_SIZE = 128
DATA_ROOT = "./data"
N_TRIALS = 100
LR = 1e-4
FINETUNE_EPOCHS = 1
SEED = 42

# --- Main Sequential Forgetting Logic ---
def run_sequential_forgetting_no_mtl(
    clients_to_forget,
    baseline_model_path,
    initial_unlearned_model_path=None,
    *,
    calculate_fisher_on: str = "digit",
):
    """
    Performs sequential unlearning on a list of clients for a no-MTL model.
    """
    print(f"--- Starting sequential forgetting for clients: {clients_to_forget} (no-MTL) ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(SEED)
    
    # If baseline model is missing, train from scratch and save to a writable location
    if not os.path.exists(baseline_model_path):
        print(f"Baseline model not found at {baseline_model_path}. Training from scratch (no-MTL)...")
        save_path = baseline_model_path
        save_dir = os.path.dirname(save_path)
        # Determine a writable path; handle read-only dirs like /kaggle/input
        if save_dir:
            if not os.path.isdir(save_dir):
                try:
                    os.makedirs(save_dir, exist_ok=True)
                except Exception as e:
                    print(f"Could not create directory {save_dir} ({e}). Saving to current directory.")
                    save_path = os.path.basename(save_path) or "baseline_all_clients_model.h5"
            elif not os.access(save_dir, os.W_OK):
                print(f"Directory {save_dir} not writable. Saving to current directory.")
                save_path = os.path.basename(save_path) or "baseline_all_clients_model.h5"
        else:
            save_path = os.path.basename(save_path) or "baseline_all_clients_model.h5"

        try:
            # For no-MTL baseline trained on all clients
            train_no_mtl_baseline_all_clients(
                dataset_name=DATASET_NAME,
                setting="non-iid",
                num_clients=NUM_CLIENTS,
                target_client_id=0,
                batch_size=BATCH_SIZE,
                num_epochs=200,
                data_root=DATA_ROOT,
                path=save_path,
                seed=SEED,
            )
            baseline_model_path = save_path
            print(f"Baseline (no-MTL) trained and saved to {baseline_model_path}")
        except Exception as e:
            print(f"Failed to train baseline (no-MTL) model: {e}")
            return

    current_model_path = baseline_model_path
    forgotten_clients = []
    
    # Generate the full client dataset structure once
    clients_data, _, full_dataset = generate_subdatasets(
        dataset_name=DATASET_NAME,
        setting="non-iid",
        num_clients=NUM_CLIENTS,
        data_root=DATA_ROOT
    )

    if DATASET_NAME == "MNIST":
        test_base = MNIST(root=DATA_ROOT, train=False, download=True, transform=transform_mnist)
    else:
        test_base = CIFAR10(root=DATA_ROOT, train=False, download=True, transform=transform_test_cifar)
    test_loader = DataLoader(test_base, batch_size=BATCH_SIZE, shuffle=False)

    for i, client_id in enumerate(clients_to_forget):
        num_forgotten = i + 1
        print(f"\n--- Stage {num_forgotten}: Forgetting client {client_id} (no-MTL) ---")

        # --- 1. SSD Unlearning ---
        unlearned_model_name = f"unlearned_model_no_mtl_forgot_{'_'.join(map(str, forgotten_clients + [client_id]))}"
        unlearned_model_path = f"{unlearned_model_name}.h5"

        if i == 0 and initial_unlearned_model_path and os.path.exists(initial_unlearned_model_path):
            print(f"Found provided unlearned model for the first client: {initial_unlearned_model_path}")
            print("Skipping initial SSD tuning.")
            unlearned_model_path = initial_unlearned_model_path
        elif os.path.exists(unlearned_model_path):
            print(f"Unlearned model for client {client_id} already exists. Skipping SSD tuning.")
        else:
            # Pass previously forgotten clients to the tuning script
            previous_forgotten_clients = [cid for cid in forgotten_clients if cid != client_id]
            previous_forgotten_clients_arg = " ".join(map(str, previous_forgotten_clients))

            tune_script = "run_ssd_tuning_no_mtl.py"
            tune_command = (
                f"python {tune_script} "
                f"--model-path {current_model_path} "
                f"--target-subset-id {client_id} "
                f"--num-forgotten-clients {num_forgotten} "
                f"--unlearned-model-name {unlearned_model_name} "
                f"--previous-forgotten-clients {previous_forgotten_clients_arg} "
                f"--current-client-id {client_id} "
                f"--baseline-variant no_mtl"
            )
            if calculate_fisher_on:
                tune_command += f" --fisher-on {calculate_fisher_on}"
            try:
                subprocess.run(tune_command, shell=True, check=True)
                print(f"--- Successfully completed SSD tuning for client {client_id} ---")
            except subprocess.CalledProcessError as e:
                print(f"--- Error running {tune_script} for client {client_id}: {e} ---")
                return

        # --- 2. Evaluation & Fine-tuning ---
        if client_id not in forgotten_clients:
            forgotten_clients.append(client_id)
        
        # Create data loaders that EXCLUDE all previously forgotten clients
        retain_indices = []
        for c_id_str, indices in clients_data.items():
            numeric_id = int(c_id_str.replace("client", "")) - 1
            if numeric_id not in forgotten_clients:
                retain_indices.extend(indices)
        
        retain_dataset = Subset(full_dataset, retain_indices)
        retain_loader = DataLoader(retain_dataset, batch_size=BATCH_SIZE, shuffle=True)

        # The 'forget_loader' should contain data for all clients forgotten so far
        forgotten_client_loaders = {}
        for cid in forgotten_clients:
            forget_indices = clients_data[f"client{cid + 1}"]
            forget_dataset = Subset(full_dataset, forget_indices)
            forgotten_client_loaders[cid] = DataLoader(forget_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Load the unlearned model
        model_to_finetune = StandardResNet(dataset_name=DATASET_NAME)
        model_to_finetune.load_state_dict(torch.load(unlearned_model_path, map_location=device))
        model_to_finetune.to(device)

        # Evaluate metrics *before* fine-tuning
        print(f"\n--- Metrics BEFORE fine-tuning (after unlearning client {client_id}) ---")
        evaluate_and_print_metrics(
            model=model_to_finetune,
            is_mtl=IS_MTL,
            retain_loader=retain_loader,
            test_loader=test_loader,
            device=device,
            forgotten_client_loaders=forgotten_client_loaders,
            current_forget_client_id=client_id
        )
        
        # Fine-tune the model
        print(f"\n--- Fine-tuning after forgetting client {client_id} (no-MTL) ---")
        finetuned_model = finetune_model(
            model=model_to_finetune,
            is_mtl=IS_MTL,
            retain_loader=retain_loader,
            forget_loader=forgotten_client_loaders[client_id], # Pass only the current forget loader
            forgotten_client_loaders=forgotten_client_loaders,  # For detailed evaluation prints
            test_loader=test_loader,
            target_client_id=client_id, # Still needed for logging inside finetune
            epochs=FINETUNE_EPOCHS,
            lr=LR,
            device=device,
            baseline_variant="no_mtl",
        )

        # Save the fine-tuned model
        finetuned_model_path = f"finetuned_model_no_mtl_forgot_{'_'.join(map(str, forgotten_clients))}.h5"
        torch.save(finetuned_model.state_dict(), finetuned_model_path)
        print(f"✓ Saved fine-tuned model to {finetuned_model_path}")
        
        current_model_path = finetuned_model_path

    print("\n--- Sequential forgetting workflow (no-MTL) completed! ---")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sequential forgetting for no-MTL with per-round baselines")
    parser.add_argument("--clients", type=int, nargs="+", default=[0, 1, 2], help="Client IDs to forget in order (rounds)")
    parser.add_argument("--baseline-model-path", type=str, default=os.environ.get("BASELINE_MODEL_PATH", "/kaggle/input/no-mtl/pytorch/default/1/baseline_all_clients_model.h5"), help="Path to the baseline no-MTL model")
    parser.add_argument("--initial-unlearned-model-path", type=str, default=os.environ.get("INITIAL_UNLEARNED_MODEL_PATH", None), help="Optional: precomputed unlearned model for the first round")
    parser.add_argument("--fisher-on", type=str, choices=["subset", "digit"], default="digit", help="Task to compute Fisher Information on during SSD (no-MTL: digit recommended)")
    args = parser.parse_args()

    run_sequential_forgetting_no_mtl(
        clients_to_forget=args.clients,
        baseline_model_path=args.baseline_model_path,
        initial_unlearned_model_path=args.initial_unlearned_model_path,
        calculate_fisher_on=args.fisher_on,
    )
