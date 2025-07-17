import random
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision.datasets import MNIST, CIFAR10

from utils import set_global_seed, SEED_DEFAULT
from data import generate_subdatasets, transform_mnist, transform_test_cifar
from training import train_single_head
from models import StandardResNet
from evaluation import (
    calculate_overall_digit_classification_accuracy,
    get_membership_attack_prob_train_only,
    calculate_digit_classification_accuracy
)


def learn_baseline_no_mtl(
    dataset_name: str = "CIFAR10",
    setting: str = "non-iid",
    num_clients: int = 10,
    excluded_client_id: int = 0,
    batch_size: int = 64,
    num_epochs: int = 10,
    data_root: str = "./data",
    path: str = "baseline_no_mtl_model.h5",
    model_class=StandardResNet,
    seed: int = SEED_DEFAULT,
    target_test_accuracy: float = None,
):
    """Train a baseline model on *all but one* client without MTL.

    The data from the `excluded_client_id` (0-indexed) is **not** used for
    training or validation. After training, the routine evaluates the model on
    the held-out client (target subset) versus the remaining clients as well as
    on the official test split. All metrics are printed for comparison.
    """

    # Setup
    assert 0 <= excluded_client_id < num_clients, (
        f"excluded_client_id must be in [0, {num_clients-1}]"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ensure reproducibility
    set_global_seed(seed)

    # Data preparation
    clients_data, _, full_dataset = generate_subdatasets(
        dataset_name=dataset_name,
        setting=setting,
        num_clients=num_clients,
        data_root=data_root,
    )

    excluded_key = f"client{excluded_client_id + 1}"
    
    included_indices = []
    for k, v in clients_data.items():
        if k != excluded_key:
            included_indices.extend(v)

    # Dataset **without** the excluded client
    included_dataset = Subset(full_dataset, included_indices)

    # Train/val split (same 80/20 rule)
    dataset_size = len(included_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    indices = list(range(dataset_size))
    random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(included_dataset, train_indices)
    val_dataset = Subset(included_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    if dataset_name == "MNIST":
        test_base = MNIST(root=data_root, train=False, download=True, transform=transform_mnist)
    else:
        test_base = CIFAR10(root=data_root, train=False, download=True, transform=transform_test_cifar)
    test_loader = DataLoader(test_base, batch_size=batch_size)

    # Model & training
    model = model_class(dataset_name=dataset_name)

    print(
        f"\n[BASELINE] Training on {num_clients-1} clients (excluding client {excluded_client_id})"
    )
    model, history = train_single_head(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        num_epochs=num_epochs,
        dataset_name=dataset_name,
        target_test_accuracy=target_test_accuracy,
    )

    torch.save(model.state_dict(), path)
    print(f"[BASELINE] Model weights saved to {path}\n")

    # Evaluation loaders
    # 1) Training data loader (contains only the samples actually seen during optimisation)
    #    Validation samples are excluded to ensure metrics are computed strictly on the training set.
    training_eval_loader = DataLoader(train_dataset, batch_size=batch_size)
    
    # 2) Target subset loader (contains only the excluded client)
    target_subset_dataset = Subset(full_dataset, clients_data[excluded_key])
    target_subset_loader = DataLoader(target_subset_dataset, batch_size=batch_size)

    # Metrics
    # Calculate accuracy on the training data (9 clients, excluding target)
    train_digit_acc = calculate_overall_digit_classification_accuracy(model, training_eval_loader, device)

    # Digit accuracy on target subset only
    target_digit_acc = calculate_overall_digit_classification_accuracy(model, target_subset_loader, device)
    
    test_digit_acc = calculate_overall_digit_classification_accuracy(model, test_loader, device)

    # Print summary
    print("[BASELINE] ---------------------------------------------")
    print(f"Digit accuracy on target subset: {target_digit_acc:.4f}")
    print(f"Digit accuracy on other subsets (train): {train_digit_acc:.4f}")
    print(f"Digit accuracy on test set: {test_digit_acc:.4f}")

    # MIA (train-only)
    mia_score = get_membership_attack_prob_train_only(training_eval_loader, target_subset_loader, model)
    print(f"Train-only MIA Accuracy (target subset vs retain): {mia_score:.2f}%")

    print("[BASELINE] ---------------------------------------------\n")

    return model, history, {
        "train_digit_tgt": target_digit_acc,
        "train_digit_oth": train_digit_acc,
        "test_digit_acc": test_digit_acc,
        "mia_score": mia_score,
    } 