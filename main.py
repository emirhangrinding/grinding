import argparse
import random
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10

from utils import set_global_seed, SEED_DEFAULT
from data import (
    generate_subdatasets, MultiTaskDataset, create_subset_data_loaders,
    transform_mnist, transform_test_cifar
)
from models import MTL_Two_Heads_ResNet
from training import learn
from dissolve import dissolve_unlearn_subset
from ssd import ssd_unlearn_subset
from retain_no_reset import retain_no_reset_unlearn_subset
from baseline import learn_baseline_excluding_client
from tuning import optimise_ssd_hyperparams
from visualization import visualize_mtl_two_heads_results
from evaluation import get_membership_attack_prob_train_only

def unlearn(
    model_path,
    target_subset_id,
    unlearning,
    gamma,
    beta,
    lr_unlearn,
    epochs_unlearn,
    model_class,
    *,
    dataset_name='CIFAR10',
    num_clients=10,
    batch_size=64,
    data_root='./data',
    finetune_task="subset",
    fine_tune_heads: bool = False,
    seed: int = SEED_DEFAULT,
    head_size: str = 'big',
    unlearning_type: str = "dissolve",
    # SSD-specific hyperparameters
    lower_bound: float = 1.0,
    exponent: float = 1.0,
    dampening_constant: float = 0.5,
    selection_weighting: float = 1.0,
):
    """
    Load the pretrained model weights and perform DISSOLVE unlearning
    with Wf zeroing and fine-tuning.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set global seeds for reproducibility
    set_global_seed(seed)

    # Generate subdatasets (needed for data loaders)
    clients_data, clients_labels, full_dataset = generate_subdatasets(
        dataset_name=dataset_name,
        setting='non-iid',           # or pass as parameter if needed
        num_clients=num_clients,
        data_root=data_root
    )

    # Create multi-task dataset
    mtl_dataset = MultiTaskDataset(full_dataset, clients_data)

    # Prepare data loaders
    dataset_size = len(mtl_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    indices = list(range(dataset_size))
    random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = torch.utils.data.Subset(mtl_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(mtl_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Create test set and loader
    # Load the official test split
    if dataset_name == 'MNIST':
        test_base = MNIST(root=data_root, train=False, download=True, transform=transform_mnist)
    else:
        test_base = CIFAR10(root=data_root, train=False, download=True, transform=transform_test_cifar)

    # Assign test samples to clients using the same class-based mapping as training
    test_class_indices = {i: [] for i in range(10)}
    for idx, (_, label) in enumerate(test_base):
        test_class_indices[label].append(idx)
    test_clients_data = {k: [] for k in clients_data.keys()}
    # Distribute test samples to clients in a round-robin fashion by class
    for class_label, idxs in test_class_indices.items():
        client_ids = [cid for cid in clients_data.keys() if any(full_dataset[i][1] == class_label for i in clients_data[cid])]
        if not client_ids:
            client_ids = list(clients_data.keys())
        for i, idx in enumerate(idxs):
            client = client_ids[i % len(client_ids)]
            test_clients_data[client].append(idx)
    test_mtl_dataset = MultiTaskDataset(test_base, test_clients_data)
    test_loader = DataLoader(test_mtl_dataset, batch_size=batch_size)

    # Create model and load saved weights
    model = model_class(dataset_name=dataset_name, num_clients=num_clients, head_size=head_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Loaded model weights from {model_path}")

    if unlearning:
        print(f"Starting unlearning for subset ID: {target_subset_id}")

        # Create retain and forget loaders
        retain_loader, forget_loader = create_subset_data_loaders(train_loader, target_subset_id)

        # MIA score BEFORE unlearning
        print("\n--- Calculating Membership Inference Attack (MIA) Accuracy (Before Unlearning) ---")
        mia_score_before = get_membership_attack_prob_train_only(retain_loader, forget_loader, model)
        print(f"Train-only MIA Accuracy: {mia_score_before:.2f}%")

        if unlearning_type.lower() == "ssd":
            # SSD Unlearning
            model = ssd_unlearn_subset(
                model,
                retain_loader,
                forget_loader,
                target_subset_id,
                device,
                lower_bound=lower_bound,
                exponent=exponent,
                dampening_constant=dampening_constant,
                selection_weighting=selection_weighting,
                test_loader=test_loader,
            )
        elif unlearning_type.lower() == "retain-no-reset":
            # Retain-No-Reset Unlearning
            model = retain_no_reset_unlearn_subset(
                model,
                retain_loader,
                forget_loader,
                target_subset_id,
                gamma,
                beta,
                lr_unlearn,
                epochs_unlearn,
                device,
                test_loader=test_loader,
                finetune_task=finetune_task,
                fine_tune_heads=fine_tune_heads,
                dataset_name=dataset_name,
                num_clients=num_clients,
                head_size=head_size,
            )
        else:
            # DeepClean / Dissolve Unlearning
            model = dissolve_unlearn_subset(
                model,
                retain_loader,
                forget_loader,
                target_subset_id,
                gamma,
                beta,
                lr_unlearn,
                epochs_unlearn,
                device,
                test_loader=test_loader,
                finetune_task=finetune_task,
                fine_tune_heads=fine_tune_heads,
                dataset_name=dataset_name,
                num_clients=num_clients,
                head_size=head_size,
            )
        print("Unlearning completed.")

        # MIA score AFTER unlearning
        print("\n--- Calculating Membership Inference Attack (MIA) Accuracy (After Unlearning) ---")
        mia_score_after = get_membership_attack_prob_train_only(retain_loader, forget_loader, model)
        print(f"Train-only MIA Accuracy: {mia_score_after:.2f}%")

    return model

def _build_cli_parser():
    """Create and return the argument parser for the script."""
    parser = argparse.ArgumentParser(
        description="Multi-Task Learning (train) and DISSOLVE Unlearning (unlearn)."
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train sub-command
    train_parser = subparsers.add_parser("train", help="Train a new model and save the weights")
    train_parser.add_argument("--dataset", default="CIFAR10", choices=["MNIST", "CIFAR10"], help="Dataset to use")
    train_parser.add_argument("--setting", default="non-iid", choices=["iid", "non-iid", "extreme-non-iid"], help="Data-partition setting")
    train_parser.add_argument("--num_clients", type=int, default=10, help="Number of clients/subsets")
    train_parser.add_argument("--batch_size", type=int, default=256, help="Mini-batch size")
    train_parser.add_argument("--num_epochs", type=int, default=200, help="Number of training epochs")
    train_parser.add_argument("--lambda_1", type=float, default=1.0, help="Weight for main task (digit classification) loss")
    train_parser.add_argument("--lambda_2", type=float, default=1.0, help="Weight for subset identification loss")
    train_parser.add_argument("--lambda_dis", type=float, default=0.1, help="Weight for disentanglement loss")
    train_parser.add_argument("--lambda_pull", type=float, default=1.0, help="Weight for pull component in disentanglement loss")
    train_parser.add_argument("--lambda_push", type=float, default=1.0, help="Weight for push component in disentanglement loss")
    train_parser.add_argument("--data_root", default="./data", help="Root directory for datasets")
    train_parser.add_argument("--head_size", default="big", choices=["big", "medium", "small"], help="Size of the classification heads: big, medium, or small")
    train_parser.add_argument("--model_path", default="model.h5", help="Path to save trained model weights")
    train_parser.add_argument("--seed", type=int, default=SEED_DEFAULT, help="Random seed for reproducibility")

    # Unlearn sub-command
    unlearn_parser = subparsers.add_parser("unlearn", help="Perform DISSOLVE unlearning on a saved model")
    unlearn_parser.add_argument("--model_path", required=True, help="Path to the pretrained model weights (e.g., model.h5)")
    unlearn_parser.add_argument("--target_subset_id", type=int, required=True, help="Subset (client) id to unlearn")
    unlearn_parser.add_argument("--gamma", type=float, default=0.1, help="Forget-sensitivity threshold or fraction")
    unlearn_parser.add_argument("--beta", type=float, default=0.1, help="Retain-sensitivity threshold or fraction")
    unlearn_parser.add_argument("--lr_unlearn", type=float, default=1e-3, help="Learning rate during unlearning fine-tuning")
    unlearn_parser.add_argument("--epochs_unlearn", type=int, default=50, help="Fine-tuning epochs during unlearning")
    unlearn_parser.add_argument("--dataset", default="CIFAR10", choices=["MNIST", "CIFAR10"], help="Dataset name (should match the model)")
    unlearn_parser.add_argument("--num_clients", type=int, default=10, help="Number of clients/subsets (should match the model)")
    unlearn_parser.add_argument("--batch_size", type=int, default=256, help="Mini-batch size for data loaders during unlearning")
    unlearn_parser.add_argument("--data_root", default="./data", help="Root directory for datasets")
    unlearn_parser.add_argument("--finetune_task", choices=["subset", "digit", "both"], default="both", help="Which task loss to use when fine-tuning")
    unlearn_parser.add_argument("--fine_tune_heads", action="store_true", help="Whether to also fine-tune the head layers in addition to ResNet")
    unlearn_parser.add_argument("--head_size", default="big", choices=["big", "medium", "small"], help="Size of the classification heads: big, medium, or small (should match the trained model)")
    unlearn_parser.add_argument("--seed", type=int, default=SEED_DEFAULT, help="Random seed for reproducibility")
    # New: Unlearning type selector
    unlearn_parser.add_argument("--unlearning_type", choices=["dissolve", "ssd", "retain-no-reset"], default="dissolve", help="Unlearning strategy: 'dissolve', 'ssd' (Selective Synaptic Dampening), or 'retain-no-reset'")

    # SSD-specific hyperparameters (exposed for advanced control)
    unlearn_parser.add_argument("--ssd_lower_bound", type=float, default=1.0, help="SSD lower_bound parameter (lambda upper cap)")
    unlearn_parser.add_argument("--ssd_exponent", type=float, default=1.0, help="SSD exponent for dampening factor")
    unlearn_parser.add_argument("--ssd_dampening_constant", type=float, default=0.5, help="SSD dampening_constant parameter")
    unlearn_parser.add_argument("--ssd_selection_weighting", type=float, default=1.0, help="SSD selection_weighting parameter")

    # Baseline sub-command
    baseline_parser = subparsers.add_parser("baseline", help="Train a baseline model excluding one client")
    baseline_parser.add_argument("--dataset", default="CIFAR10", choices=["MNIST", "CIFAR10"], help="Dataset to use")
    baseline_parser.add_argument("--setting", default="non-iid", choices=["iid", "non-iid", "extreme-non-iid"], help="Data-partition setting")
    baseline_parser.add_argument("--num_clients", type=int, default=10, help="Number of clients/subsets")
    baseline_parser.add_argument("--excluded_client_id", type=int, required=True, help="Client ID to exclude from training (0-indexed)")
    baseline_parser.add_argument("--batch_size", type=int, default=64, help="Mini-batch size")
    baseline_parser.add_argument("--num_epochs", type=int, default=200, help="Number of training epochs")
    baseline_parser.add_argument("--lambda_1", type=float, default=1.0, help="Weight for main task (digit classification) loss")
    baseline_parser.add_argument("--lambda_2", type=float, default=1.0, help="Weight for subset identification loss")
    baseline_parser.add_argument("--lambda_dis", type=float, default=0.1, help="Weight for disentanglement loss")
    baseline_parser.add_argument("--lambda_pull", type=float, default=1.0, help="Weight for pull component in disentanglement loss")
    baseline_parser.add_argument("--lambda_push", type=float, default=1.0, help="Weight for push component in disentanglement loss")
    baseline_parser.add_argument("--data_root", default="./data", help="Root directory for datasets")
    baseline_parser.add_argument("--head_size", default="big", choices=["big", "medium", "small"], help="Size of the classification heads")
    baseline_parser.add_argument("--model_path", default="baseline_model.h5", help="Path to save trained model weights")
    baseline_parser.add_argument("--seed", type=int, default=SEED_DEFAULT, help="Random seed for reproducibility")

    # Tune-SSD sub-command
    tune_parser = subparsers.add_parser("tune_ssd", help="Hyper-parameter tuning for SSD via Optuna")
    tune_parser.add_argument("--model_path", required=True, help="Path to the pretrained model weights (e.g., model.h5)")
    tune_parser.add_argument("--target_subset_id", type=int, required=True, help="Subset (client) id to unlearn during tuning")
    tune_parser.add_argument("--n_trials", type=int, default=25, help="Number of Optuna trials")
    tune_parser.add_argument("--dataset", default="CIFAR10", choices=["MNIST", "CIFAR10"], help="Dataset name (should match the model)")
    tune_parser.add_argument("--num_clients", type=int, default=10, help="Number of clients/subsets (should match the model)")
    tune_parser.add_argument("--batch_size", type=int, default=256, help="Mini-batch size for data loaders during tuning")
    tune_parser.add_argument("--data_root", default="./data", help="Root directory for datasets")
    tune_parser.add_argument("--head_size", default="big", choices=["big", "medium", "small"], help="Size of the classification heads: big, medium, or small (should match the trained model)")
    tune_parser.add_argument("--seed", type=int, default=SEED_DEFAULT, help="Random seed for reproducibility")

    return parser

def main():
    parser = _build_cli_parser()
    args = parser.parse_args()

    if args.command == "train":
        # Training
        model, history, client_labels = learn(
            dataset_name=args.dataset,
            setting=args.setting,
            num_clients=args.num_clients,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            lambda_1=args.lambda_1,
            lambda_2=args.lambda_2,
            lambda_dis=args.lambda_dis,
            lambda_pull=args.lambda_pull,
            lambda_push=args.lambda_push,
            data_root=args.data_root,
            path=args.model_path,
            model_class=MTL_Two_Heads_ResNet,
            seed=args.seed,
            head_size=args.head_size,
        )

        visualize_mtl_two_heads_results(history)
        print("\nClient Label Distribution:")
        for client, labels in client_labels.items():
            print(f"{client}: {labels}")

    elif args.command == "unlearn":
        # Unlearning
        _ = unlearn(
            model_path=args.model_path,
            target_subset_id=args.target_subset_id,
            unlearning=True,
            gamma=args.gamma,
            beta=args.beta,
            lr_unlearn=args.lr_unlearn,
            epochs_unlearn=args.epochs_unlearn,
            model_class=MTL_Two_Heads_ResNet,
            dataset_name=args.dataset,
            num_clients=args.num_clients,
            batch_size=args.batch_size,
            data_root=args.data_root,
            finetune_task=args.finetune_task,
            fine_tune_heads=args.fine_tune_heads,
            seed=args.seed,
            head_size=args.head_size,
            unlearning_type=args.unlearning_type,
            lower_bound=args.ssd_lower_bound,
            exponent=args.ssd_exponent,
            dampening_constant=args.ssd_dampening_constant,
            selection_weighting=args.ssd_selection_weighting,
        )

    elif args.command == "baseline":
        # Baseline Training
        model, history, metrics = learn_baseline_excluding_client(
            dataset_name=args.dataset,
            setting=args.setting,
            num_clients=args.num_clients,
            excluded_client_id=args.excluded_client_id,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            lambda_1=args.lambda_1,
            lambda_2=args.lambda_2,
            lambda_dis=args.lambda_dis,
            lambda_pull=args.lambda_pull,
            lambda_push=args.lambda_push,
            data_root=args.data_root,
            path=args.model_path,
            seed=args.seed,
            head_size=args.head_size,
        )

        print("\nBaseline Training Results:")
        print(f"Final metrics: {metrics}")

    elif args.command == "tune_ssd":
        # SSD Hyper-parameter Tuning
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_global_seed(args.seed)

        # Recreate data loaders (similar to unlearn)
        clients_data, _clients_labels, full_dataset = generate_subdatasets(
            dataset_name=args.dataset,
            setting="non-iid",
            num_clients=args.num_clients,
            data_root=args.data_root,
        )

        mtl_dataset = MultiTaskDataset(full_dataset, clients_data)

        # Train/Val split for loaders (we only need train here)
        dataset_size = len(mtl_dataset)
        train_size = int(0.8 * dataset_size)
        indices = list(range(dataset_size))
        random.shuffle(indices)
        train_indices = indices[:train_size]

        train_dataset = torch.utils.data.Subset(mtl_dataset, train_indices)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        retain_loader, forget_loader = create_subset_data_loaders(train_loader, args.target_subset_id)

        # Test loader (official test split adapted to clients)
        if args.dataset == "MNIST":
            test_base = MNIST(root=args.data_root, train=False, download=True, transform=transform_mnist)
        else:
            test_base = CIFAR10(root=args.data_root, train=False, download=True, transform=transform_test_cifar)

        test_clients_data = {k: [] for k in clients_data.keys()}
        test_class_indices = {i: [] for i in range(10)}
        for idx, (_, label) in enumerate(test_base):
            test_class_indices[label].append(idx)
        for class_label, idxs in test_class_indices.items():
            client_ids = [cid for cid in clients_data.keys() if any(full_dataset[i][1] == class_label for i in clients_data[cid])]
            if not client_ids:
                client_ids = list(clients_data.keys())
            for i, idx in enumerate(idxs):
                client = client_ids[i % len(client_ids)]
                test_clients_data[client].append(idx)

        test_dataset = MultiTaskDataset(test_base, test_clients_data)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

        # Load pretrained model
        model = MTL_Two_Heads_ResNet(dataset_name=args.dataset, num_clients=args.num_clients, head_size=args.head_size)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.to(device)
        model.eval()

        _ = optimise_ssd_hyperparams(
            pretrained_model=model,
            retain_loader=retain_loader,
            forget_loader=forget_loader,
            test_loader=test_loader,
            device=device,
            target_subset_id=args.target_subset_id,
            n_trials=args.n_trials,
            seed=args.seed,
        )

if __name__ == "__main__":
    main() 