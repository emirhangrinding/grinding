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
    *,
    baseline_variant: str = None,
    kill_output_neuron: bool = True,
    digit_metrics_only: bool = False,
    calculate_fisher_on: str = "subset",
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

    # Infer baseline variant if not provided
    if baseline_variant is None:
        baseline_variant = "mtl_ce" if ("no_dis" in os.path.basename(baseline_model_path)) else "mtl"

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
                f"--current-client-id {client_id} "
                f"--baseline-variant {baseline_variant} "
                f"--fisher-on {calculate_fisher_on}"
            )
            if digit_metrics_only:
                tune_command += " --digit-metrics-only"
            if kill_output_neuron:
                tune_command += " --kill-output-neuron"
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
            lambda_subset=0.0,
            device=device,
            baseline_variant=baseline_variant,
        )

        # Save the fine-tuned model, making it the input for the next round
        finetuned_model_path = f"finetuned_model_mtl_forgot_{'_'.join(map(str, forgotten_clients))}.h5"
        torch.save(finetuned_model.state_dict(), finetuned_model_path)
        print(f"✓ Saved fine-tuned model to {finetuned_model_path}")
        
        current_model_path = finetuned_model_path

    print("\n--- Sequential forgetting workflow completed! ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sequential forgetting for MTL with per-round baselines")
    parser.add_argument("--clients", type=int, nargs="+", default=[0, 1, 2], help="Client IDs to forget in order (rounds)")
    parser.add_argument("--baseline-model-path", type=str, default=os.environ.get("BASELINE_MODEL_PATH", "/kaggle/input/mtl/pytorch/default/1/baseline_mtl_all_clients.h5"), help="Path to the baseline MTL model")
    parser.add_argument("--initial-unlearned-model-path", type=str, default=os.environ.get("INITIAL_UNLEARNED_MODEL_PATH", None), help="Optional: precomputed unlearned model for the first round")
    parser.add_argument("--baseline-variant", type=str, choices=["mtl", "mtl_ce"], default=None, help="Baseline variant to use (mtl or mtl_ce)")
    parser.add_argument("--lambda-digit", dest="lambda_digit", type=float, default=None, help="Weight for adversarial digit loss (default 0.3)")
    parser.add_argument("--digit-metrics-only", action="store_true", help="Use only digit metrics during SSD tuning objective")
    parser.add_argument("--fisher-on", type=str, choices=["subset", "digit"], default="subset", help="Task to compute Fisher Information on during SSD")
    parser.add_argument("--kill-output-neuron", action="store_true", help="Suppress the target subset's output neuron during evaluation after SSD")
    args = parser.parse_args()

    resolved_lambda_digit = args.lambda_digit if args.lambda_digit is not None else float(os.environ.get("LAMBDA_DIGIT", "0.3"))

    run_sequential_forgetting(
        clients_to_forget=args.clients,
        baseline_model_path=args.baseline_model_path,
        initial_unlearned_model_path=args.initial_unlearned_model_path,
        lambda_digit=resolved_lambda_digit,
        lambda_subset=0.0,
        baseline_variant=args.baseline_variant,
        kill_output_neuron=args.kill_output_neuron or True,
        digit_metrics_only=args.digit_metrics_only,
        calculate_fisher_on=args.fisher_on,
    )
