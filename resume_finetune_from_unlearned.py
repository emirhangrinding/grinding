#!/usr/bin/env python3
import os
import argparse
import torch
from torch.utils.data import DataLoader, Subset

from data import generate_subdatasets, MultiTaskDataset, transform_test_cifar
from models import MTL_Two_Heads_ResNet
from finetune import finetune_model
from evaluation import evaluate_and_print_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume: fine-tune from an existing unlearned MTL model.")
    parser.add_argument("--unlearned-model-path", type=str, required=True, help="Path to unlearned model .h5 (e.g., unlearned_model_mtl_forgot_0_1_2.h5)")
    parser.add_argument("--forgotten-clients", type=int, nargs="+", default=[0, 1, 2], help="List of forgotten client IDs (0-indexed). Default: 0 1 2")
    parser.add_argument("--target-client-id", type=int, default=2, help="Client ID to fine-tune against (0-indexed). Default: 2")
    parser.add_argument("--dataset-name", type=str, choices=["MNIST", "CIFAR10", "CIFAR100"], default="CIFAR100")
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--head-size", type=str, choices=["big", "medium", "small"], default="medium")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda-digit", dest="lambda_digit", type=float, default=None, help="Adversarial digit loss weight. Defaults to env LAMBDA_DIGIT or 0.3")
    parser.add_argument("--baseline-variant", type=str, choices=["mtl", "mtl_ce"], default="mtl")
    parser.add_argument("--kill-output-neuron", action="store_true", help="Suppress forgotten clients' subset logits during eval/finetune")
    parser.add_argument("--no-grid", action="store_true", help="Disable grid search; run a single fine-tune with given epochs and lambda-digit")
    parser.add_argument("--lambda-digit-grid", type=float, nargs="+", default=None, help="Grid for lambda_digit (space-separated floats)")
    parser.add_argument("--lambda-subset-grid", type=float, nargs="+", default=None, help="Grid for lambda_subset (space-separated floats)")
    parser.add_argument("--epochs-grid", type=int, nargs="+", default=None, help="Grid for epochs (space-separated ints)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unlearned_model_path = os.path.expanduser(args.unlearned_model_path)
    if not os.path.exists(unlearned_model_path):
        raise FileNotFoundError(f"Unlearned model not found: {unlearned_model_path}")

    lambda_digit = args.lambda_digit if args.lambda_digit is not None else float(os.environ.get("LAMBDA_DIGIT", "0.3"))

    # Build full dataset and client splits
    clients_data, _, full_dataset = generate_subdatasets(
        dataset_name=args.dataset_name,
        setting="non-iid",
        num_clients=args.num_clients,
        data_root=args.data_root,
    )

    # Test loader
    if args.dataset_name == "MNIST":
        from data import transform_mnist
        from torchvision.datasets import MNIST
        test_base = MNIST(root=args.data_root, train=False, download=True, transform=transform_mnist)
    elif args.dataset_name == "CIFAR10":
        from torchvision.datasets import CIFAR10
        test_base = CIFAR10(root=args.data_root, train=False, download=True, transform=transform_test_cifar)
    else:
        from torchvision.datasets import CIFAR100
        test_base = CIFAR100(root=args.data_root, train=False, download=True, transform=transform_test_cifar)
    test_loader = DataLoader(test_base, batch_size=args.batch_size, shuffle=False)

    # Multi-task dataset mapping
    mtl_dataset = MultiTaskDataset(full_dataset, clients_data)
    dsidx_to_mtlidx = {ds_idx: pos for pos, ds_idx in enumerate(mtl_dataset.valid_indices)}

    # Per-client indices in MTL index space
    client_to_mtl_indices = {}
    for c_id, ds_indices in clients_data.items():
        mtl_indices = [dsidx_to_mtlidx[ds_idx] for ds_idx in ds_indices if ds_idx in dsidx_to_mtlidx]
        client_to_mtl_indices[c_id] = mtl_indices

    # Build retain loader: exclude all forgotten clients
    forgotten_set = set(args.forgotten_clients)
    retain_mtl_indices = []
    for c_id in clients_data.keys():
        numeric_id = int(c_id.replace("client", "")) - 1
        if numeric_id not in forgotten_set:
            retain_mtl_indices.extend(client_to_mtl_indices[c_id])
    retain_dataset = Subset(mtl_dataset, retain_mtl_indices)
    retain_loader = DataLoader(retain_dataset, batch_size=args.batch_size, shuffle=True)

    # Build forgotten client loaders (for metrics and the current forget_loader)
    forgotten_client_loaders = {}
    for cid in sorted(forgotten_set):
        forget_mtl_indices = client_to_mtl_indices[f"client{cid + 1}"]
        forget_dataset = Subset(mtl_dataset, forget_mtl_indices)
        forgotten_client_loaders[cid] = DataLoader(forget_dataset, batch_size=args.batch_size, shuffle=False)

    # Model: load unlearned weights
    model = MTL_Two_Heads_ResNet(dataset_name=args.dataset_name, num_clients=args.num_clients, head_size=args.head_size)
    state = torch.load(unlearned_model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)

    # Ensure subset neurons for all forgotten clients are suppressed
    if hasattr(model, "kill_output_neuron"):
        model.kill_output_neuron = True if args.kill_output_neuron else False
        if hasattr(model, "killed_subset_ids"):
            model.killed_subset_ids = set(forgotten_set)
        elif hasattr(model, "killed_subset_id"):
            model.killed_subset_id = int(args.target_client_id)

    # Optional: print metrics before fine-tuning
    print("\n--- Metrics BEFORE fine-tuning (resume) ---")
    evaluate_and_print_metrics(
        model=model,
        is_mtl=True,
        retain_loader=retain_loader,
        test_loader=test_loader,
        device=device,
        forgotten_client_loaders=forgotten_client_loaders,
        current_forget_client_id=args.target_client_id,
        ssd_print_style=True,
    )

    # Fine-tune
    print(f"\n--- Fine-tuning resumed for client {args.target_client_id} ---")
    out_path = f"finetuned_model_mtl_forgot_{'_'.join(map(str, sorted(forgotten_set)))}.h5"
    finetuned_model = finetune_model(
        model=model,
        is_mtl=True,
        retain_loader=retain_loader,
        forget_loader=forgotten_client_loaders[args.target_client_id],
        forgotten_client_loaders=forgotten_client_loaders,
        test_loader=test_loader,
        target_client_id=args.target_client_id,
        epochs=args.epochs,
        lr=args.lr,
        lambda_digit=lambda_digit,
        lambda_subset=0.0,
        device=device,
        baseline_variant=args.baseline_variant,
        search_lambdas=(not args.no_grid),
        lambda_digit_grid=args.lambda_digit_grid,
        lambda_subset_grid=args.lambda_subset_grid,
        epochs_grid=args.epochs_grid,
        save_best_model_path=out_path,
    )

    # Save (already saved inside fine-tuner when grid is enabled, but keep a final save to be safe)
    try:
        torch.save(finetuned_model.state_dict(), out_path)
    except Exception:
        pass
    print(f"\n✓ Saved fine-tuned model to {out_path}")


if __name__ == "__main__":
    main()


