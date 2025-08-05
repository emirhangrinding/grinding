#!/usr/bin/env python3
import subprocess
import os
import torch
from torch.utils.data import DataLoader, Subset

from utils import set_global_seed, SEED_DEFAULT
from data import generate_subdatasets
from models import StandardResNet
from finetune import finetune_model
from torchvision.datasets import MNIST, CIFAR10
from data import transform_mnist, transform_test_cifar

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
):
    """
    Performs sequential unlearning on a list of clients for a no-MTL model.
    """
    print(f"--- Starting sequential forgetting for clients: {clients_to_forget} (no-MTL) ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(SEED)
    
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
            tune_script = "run_ssd_tuning_no_mtl.py"
            tune_command = (
                f"python {tune_script} "
                f"--model-path {current_model_path} "
                f"--target-subset-id {client_id} "
                f"--num-forgotten-clients {num_forgotten} "
                f"--unlearned-model-name {unlearned_model_name}"
            )
            try:
                subprocess.run(tune_command, shell=True, check=True)
                print(f"--- Successfully completed SSD tuning for client {client_id} ---")
            except subprocess.CalledProcessError as e:
                print(f"--- Error running {tune_script} for client {client_id}: {e} ---")
                return

        # --- 2. Fine-tuning ---
        print(f"\n--- Fine-tuning after forgetting client {client_id} (no-MTL) ---")

        forgotten_clients.append(client_id)
        
        # Create data loaders that EXCLUDE all previously forgotten clients
        retain_indices = []
        for c_id_str, indices in clients_data.items():
            numeric_id = int(c_id_str.replace("client", "")) - 1
            if numeric_id not in forgotten_clients:
                retain_indices.extend(indices)
        
        retain_dataset = Subset(full_dataset, retain_indices)
        retain_loader = DataLoader(retain_dataset, batch_size=BATCH_SIZE, shuffle=True)

        # The 'forget_loader' for fine-tuning should be for the client *just* forgotten
        forget_indices = clients_data[f"client{client_id + 1}"]
        forget_dataset = Subset(full_dataset, forget_indices)
        forget_loader = DataLoader(forget_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Load the unlearned model
        model_to_finetune = StandardResNet(dataset_name=DATASET_NAME)
        model_to_finetune.load_state_dict(torch.load(unlearned_model_path, map_location=device))
        
        # Fine-tune the model
        finetuned_model = finetune_model(
            model=model_to_finetune,
            is_mtl=IS_MTL,
            retain_loader=retain_loader,
            forget_loader=forget_loader,
            test_loader=test_loader,
            target_client_id=client_id, # Still needed for logging inside finetune
            epochs=FINETUNE_EPOCHS,
            lr=LR,
            device=device,
        )

        # Save the fine-tuned model
        finetuned_model_path = f"finetuned_model_no_mtl_forgot_{'_'.join(map(str, forgotten_clients))}.h5"
        torch.save(finetuned_model.state_dict(), finetuned_model_path)
        print(f"✓ Saved fine-tuned model to {finetuned_model_path}")
        
        current_model_path = finetuned_model_path

    print("\n--- Sequential forgetting workflow (no-MTL) completed! ---")

if __name__ == "__main__":
    clients_to_forget_seq = [0, 1] 
    
    baseline_model = "/kaggle/input/no-mtl/pytorch/default/1/baseline_all_clients_model.h5"
    initial_unlearned_model = "/kaggle/input/unlearned/pytorch/default/1/unlearned_model_no_mtl.h5"


    run_sequential_forgetting_no_mtl(
        clients_to_forget_seq, 
        baseline_model,
        initial_unlearned_model_path=initial_unlearned_model
    )
