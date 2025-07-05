import random
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST, CIFAR10

from utils import set_global_seed, SEED_DEFAULT
from data import generate_subdatasets, MultiTaskDataset, transform_mnist, transform_test_cifar
from training import train_mtl_two_heads
from models import MTL_Two_Heads_ResNet
from evaluation import (
    calculate_digit_classification_accuracy,
    calculate_subset_identification_accuracy,
    calculate_overall_digit_classification_accuracy,
    calculate_overall_subset_identification_accuracy,
    get_membership_attack_prob_train_only
)

def learn_baseline_excluding_client(
    dataset_name: str = "CIFAR10",
    setting: str = "non-iid",
    num_clients: int = 10,
    excluded_client_id: int = 0,
    batch_size: int = 64,
    num_epochs: int = 10,
    lambda_1: float = 1.0,
    lambda_2: float = 1.0,
    lambda_dis: float = 0.1,
    lambda_pull: float = 1.0,
    lambda_push: float = 1.0,
    data_root: str = "./data",
    path: str = "baseline_model.h5",
    model_class=MTL_Two_Heads_ResNet,
    seed: int = SEED_DEFAULT,
    head_size: str = "big",
):
    """Train a baseline model on *all but one* client.

    The data from the `excluded_client_id` (0-indexed) is **not** used for
    training or validation.  After training, the routine evaluates the model on
    the held-out client (target subset) versus the remaining clients as well as
    on the official test split.  All metrics are printed in a format similar to
    the unlearning pipeline for easy side-by-side comparison.
    """

    # Setup
    assert 0 <= excluded_client_id < num_clients, (
        f"excluded_client_id must be in [0, {num_clients-1}]"
    )
    target_subset_id = excluded_client_id  # naming consistency

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ensure reproducibility
    set_global_seed(seed)

    # Data preparation
    clients_data, clients_labels, full_dataset = generate_subdatasets(
        dataset_name=dataset_name,
        setting=setting,
        num_clients=num_clients,
        data_root=data_root,
    )

    excluded_key = f"client{excluded_client_id + 1}"
    included_clients_data = {k: v for k, v in clients_data.items() if k != excluded_key}

    # Dataset **without** the excluded client
    mtl_dataset_included = MultiTaskDataset(full_dataset, included_clients_data)

    # Train/val split (same 80/20 rule)
    dataset_size = len(mtl_dataset_included)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    indices = list(range(dataset_size))
    random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(mtl_dataset_included, train_indices)
    val_dataset = Subset(mtl_dataset_included, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Model & training
    model = model_class(dataset_name=dataset_name, num_clients=num_clients, head_size=head_size)

    print(
        f"\n[BASELINE] Training on {num_clients-1} clients (excluding client {excluded_client_id})"
    )
    model, history = train_mtl_two_heads(
        model=model,
        train_loader=train_loader,
        test_loader=eval_loader,
        device=device,
        num_epochs=num_epochs,
        lambda_1=lambda_1,
        lambda_2=lambda_2,
        lambda_dis=lambda_dis,
        lambda_pull=lambda_pull,
        lambda_push=lambda_push,
        dataset_name=dataset_name,
    )

    torch.save(model.state_dict(), path)
    print(f"[BASELINE] Model weights saved to {path}\n")

    # Evaluation loaders
    # 1) Training data loader (contains only the 9 clients used for training)
    training_eval_loader = DataLoader(mtl_dataset_included, batch_size=batch_size)
    
    # 2) Target subset loader (contains only the excluded client)
    excluded_clients_data = {excluded_key: clients_data[excluded_key]}
    target_subset_dataset = MultiTaskDataset(full_dataset, excluded_clients_data)
    target_subset_loader = DataLoader(target_subset_dataset, batch_size=batch_size)

    # 3) Test loader (official test split mapped to clients)
    if dataset_name == "MNIST":
        test_base = MNIST(root=data_root, train=False, download=True, transform=transform_mnist)
    else:
        test_base = CIFAR10(root=data_root, train=False, download=True, transform=transform_test_cifar)

    # Replicate the class-aware distribution used earlier
    test_class_indices = {i: [] for i in range(10)}
    for idx, (_, lbl) in enumerate(test_base):
        test_class_indices[lbl].append(idx)

    test_clients_data = {k: [] for k in clients_data.keys()}
    for class_label, idxs in test_class_indices.items():
        client_ids = [cid for cid in clients_data.keys() if any(full_dataset[i][1] == class_label for i in clients_data[cid])]
        if not client_ids:
            client_ids = list(clients_data.keys())
        for i, idx in enumerate(idxs):
            client = client_ids[i % len(client_ids)]
            test_clients_data[client].append(idx)

    test_mtl_dataset = MultiTaskDataset(test_base, test_clients_data)
    test_loader = DataLoader(test_mtl_dataset, batch_size=batch_size)

    # Metrics
    # Calculate accuracy on the training data (9 clients, excluding target)
    train_digit_acc = calculate_overall_digit_classification_accuracy(model, training_eval_loader, device)
    train_subset_acc = calculate_overall_subset_identification_accuracy(model, training_eval_loader, device)
    
    # Calculate accuracy on the target subset only
    target_digit_acc = calculate_overall_digit_classification_accuracy(model, target_subset_loader, device)
    target_subset_acc = calculate_overall_subset_identification_accuracy(model, target_subset_loader, device)

    # Use the individual values for compatibility
    tgt_acc = target_digit_acc
    oth_acc = train_digit_acc
    sub_tgt_acc = target_subset_acc
    sub_oth_acc = train_subset_acc

    test_digit_tgt, test_digit_oth = calculate_digit_classification_accuracy(model, test_loader, device, target_subset_id)
    test_digit_acc = calculate_overall_digit_classification_accuracy(model, test_loader, device)

    # Print summary
    print("[BASELINE] ---------------------------------------------")
    print(f"Digit accuracy on target subset: {tgt_acc:.4f}")
    print(f"Digit accuracy on other subsets: {oth_acc:.4f}")
    print(f"Subset ID accuracy on target subset: {sub_tgt_acc:.4f}")
    print(f"Subset ID accuracy on other subsets: {sub_oth_acc:.4f}")
    print("    [TEST] Digit accuracy: {:.4f}".format(test_digit_acc))

    # MIA (train-only)
    mia_score = get_membership_attack_prob_train_only(training_eval_loader, target_subset_loader, model)
    print(f"Train-only MIA Accuracy (target subset vs retain): {mia_score:.2f}%")

    print("[BASELINE] ---------------------------------------------\n")

    return model, history, {
        "train_digit_tgt": tgt_acc,
        "train_digit_oth": oth_acc,
        "train_subset_tgt": sub_tgt_acc,
        "train_subset_oth": sub_oth_acc,
        "test_digit_tgt": test_digit_tgt,
        "test_digit_oth": test_digit_oth,
        "mia_score": mia_score,
    } 