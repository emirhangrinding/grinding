#!/usr/bin/env python3
import subprocess
import os
import argparse
import torch
from torch.utils.data import DataLoader, Subset

from utils import set_global_seed, SEED_DEFAULT
from data import generate_subdatasets, MultiTaskDataset, create_subset_data_loaders
from models import MTL_Two_Heads_ResNet
from finetune import finetune_model
from evaluation import evaluate_and_print_metrics
from torchvision.datasets import MNIST, CIFAR10
from data import transform_mnist, transform_test_cifar

# --- Configuration ---
IS_MTL = True
DATASET_NAME = "CIFAR10"
NUM_CLIENTS = 10
BATCH_SIZE = 128
DATA_ROOT = "./data"
HEAD_SIZE = "medium"
N_TRIALS = 100 
LR = 1e-4
FINETUNE_EPOCHS = 1
SEED = 42

# --- Main Sequential Forgetting Logic ---
def run_sequential_forgetting(
    clients_to_forget,
    baseline_model_path,
    initial_unlearned_model_path=None,
    initial_forgotten_clients=None,
    override_unlearned_model_path=None,
    lambda_digit: float = 0.3,
    lambda_subset: float = 0.2,
):
    """
    Performs sequential unlearning on a list of clients.
    """
    print(f"--- Starting sequential forgetting for clients: {clients_to_forget} ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(SEED)
    
    current_model_path = baseline_model_path
    # Allows starting from a later round by seeding previously forgotten clients
    forgotten_clients = list(initial_forgotten_clients) if initial_forgotten_clients else []
    
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
        print(f"\n--- Stage {num_forgotten}: Forgetting client {client_id} ---")

        # --- 1. SSD Unlearning ---
        unlearned_model_name = f"unlearned_model_mtl_forgot_{'_'.join(map(str, forgotten_clients + [client_id]))}"
        unlearned_model_path = f"{unlearned_model_name}.h5"

        # Check for a pre-existing model for the *first* client, an override path, or a locally saved one
        if i == 0 and initial_unlearned_model_path and os.path.exists(initial_unlearned_model_path):
            print(f"Found provided unlearned model for the first client: {initial_unlearned_model_path}")
            print("Skipping initial SSD tuning.")
            unlearned_model_path = initial_unlearned_model_path
        elif override_unlearned_model_path and os.path.exists(override_unlearned_model_path):
            print(f"Using provided unlearned model override: {override_unlearned_model_path}")
            print("Skipping SSD tuning for this stage.")
            unlearned_model_path = override_unlearned_model_path
        elif os.path.exists(unlearned_model_path):
            print(f"Unlearned model for client {client_id} already exists locally. Skipping SSD tuning.")
        else:
            # Pass previously forgotten clients to the tuning script
            previous_forgotten_clients = [cid for cid in forgotten_clients if cid != client_id]
            previous_forgotten_clients_arg = " ".join(map(str, previous_forgotten_clients))

            tune_script = "run_ssd_tuning.py"
            tune_command = (
                f"python {tune_script} "
                f"--model-path {current_model_path} "
                f"--target-subset-id {client_id} "
                f"--num-forgotten-clients {num_forgotten} "
                f"--unlearned-model-name {unlearned_model_name} "
                f"--previous-forgotten-clients {previous_forgotten_clients_arg} "
                f"--current-client-id {client_id}"
            )
            try:
                subprocess.run(tune_command, shell=True, check=True)
                print(f"--- Successfully completed SSD tuning for client {client_id} ---")
            except subprocess.CalledProcessError as e:
                print(f"--- Error running {tune_script} for client {client_id}: {e} ---")
                return

        # --- 2. Evaluation & Fine-tuning ---
        
        # Update the list of forgotten clients for the next stage
        # This is done *before* creating dataloaders to ensure correct data exclusion
        if client_id not in forgotten_clients:
            forgotten_clients.append(client_id)
        
        # Create data loaders that EXCLUDE all previously forgotten clients
        all_indices = []
        for c_id, indices in clients_data.items():
            numeric_id = int(c_id.replace("client", "")) - 1
            if numeric_id not in forgotten_clients:
                 all_indices.extend(indices)
        
        mtl_dataset = MultiTaskDataset(full_dataset, clients_data)
        
        retain_dataset = Subset(mtl_dataset, all_indices)
        retain_loader = DataLoader(retain_dataset, batch_size=BATCH_SIZE, shuffle=True)

        # The 'forget_loader' should contain data for all clients forgotten so far
        forgotten_client_loaders = {}
        for cid in forgotten_clients:
            forget_indices = clients_data[f"client{cid + 1}"]
            forget_dataset = Subset(mtl_dataset, forget_indices)
            forgotten_client_loaders[cid] = DataLoader(forget_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Load the unlearned model
        model_to_finetune = MTL_Two_Heads_ResNet(dataset_name=DATASET_NAME, num_clients=NUM_CLIENTS, head_size=HEAD_SIZE)
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
        print(f"\n--- Fine-tuning after forgetting client {client_id} ---")
        finetuned_model = finetune_model(
            model=model_to_finetune,
            is_mtl=IS_MTL,
            retain_loader=retain_loader,
            forget_loader=forgotten_client_loaders[client_id],  # Pass only the current forget loader
            forgotten_client_loaders=forgotten_client_loaders,  # For detailed evaluation prints
            test_loader=test_loader,
            target_client_id=client_id,
            epochs=FINETUNE_EPOCHS,
            lr=LR,
            lambda_digit=lambda_digit,
            lambda_subset=lambda_subset,
            device=device,
        )

        # Save the fine-tuned model, making it the input for the next round
        finetuned_model_path = f"finetuned_model_mtl_forgot_{'_'.join(map(str, forgotten_clients))}.h5"
        torch.save(finetuned_model.state_dict(), finetuned_model_path)
        print(f"✓ Saved fine-tuned model to {finetuned_model_path}")
        
        current_model_path = finetuned_model_path

    print("\n--- Sequential forgetting workflow completed! ---")

if __name__ == "__main__":
    # CLI for lambda overrides with env fallbacks
    parser = argparse.ArgumentParser(description="Sequential forgetting with optional lambda overrides")
    parser.add_argument("--lambda-digit", dest="lambda_digit", type=float, default=None, help="Weight for adversarial digit loss")
    parser.add_argument("--lambda-subset", dest="lambda_subset", type=float, default=None, help="Weight for adversarial subset loss")
    # Parse known args only to avoid interfering with other external arg parsers
    args, _ = parser.parse_known_args()

    # Default paths (can be overridden via environment variables below)
    baseline_model = os.environ.get(
        "BASELINE_MODEL_PATH",
        "/kaggle/input/mtl/pytorch/default/1/baseline_mtl_all_clients.h5",
    )
    initial_unlearned_model = os.environ.get(
        "INITIAL_UNLEARNED_MODEL_PATH",
        "/kaggle/input/unlearned/pytorch/default/1/unlearned_model_mtl.h5",
    )

    # Resolve lambdas from CLI or env (env var names match CLI flags but uppercase)
    resolved_lambda_digit = args.lambda_digit if args.lambda_digit is not None else float(os.environ.get("LAMBDA_DIGIT", "0.3"))
    resolved_lambda_subset = args.lambda_subset if args.lambda_subset is not None else float(os.environ.get("LAMBDA_SUBSET", "0.2"))

    print(f"Using lambda_digit={resolved_lambda_digit}, lambda_subset={resolved_lambda_subset}")

    # Support a simple round-2-only finetuning flow via environment variables
    # Set ROUND2_ONLY=1 and provide ROUND2_UNLEARNED_MODEL_PATH to skip SSD and only fine-tune.
    if os.environ.get("ROUND2_ONLY", "0") == "1":
        round2_path = os.environ.get("ROUND2_UNLEARNED_MODEL_PATH")
        if not round2_path or not os.path.exists(round2_path):
            raise FileNotFoundError(
                "ROUND2_ONLY is set but ROUND2_UNLEARNED_MODEL_PATH is missing or does not exist."
            )
        # Previously forgotten client 0; now fine-tune after forgetting client 1 as well
        run_sequential_forgetting(
            clients_to_forget=[1],
            baseline_model_path=baseline_model,
            initial_unlearned_model_path=None,
            initial_forgotten_clients=[0],
            override_unlearned_model_path=round2_path,
            lambda_digit=resolved_lambda_digit,
            lambda_subset=resolved_lambda_subset,
        )
    else:
        # Full two-round flow by default
        run_sequential_forgetting(
            clients_to_forget=[0, 1],
            baseline_model_path=baseline_model,
            initial_unlearned_model_path=initial_unlearned_model,
            lambda_digit=resolved_lambda_digit,
            lambda_subset=resolved_lambda_subset,
        )
