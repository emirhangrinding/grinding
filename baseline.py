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
    calculate_subset_identification_accuracy_multiple_targets,
    calculate_overall_digit_classification_accuracy,
    get_membership_attack_prob_train_only,
)

def learn_baseline_excluding_clients(
    dataset_name: str = "CIFAR10",
    setting: str = "non-iid",
    num_clients: int = 10,
    excluded_client_ids: list[int] = [0],
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
    """Train a baseline model on all but the specified clients.

    The data from the `excluded_client_ids` is **not** used for
    training or validation. After training, the routine evaluates the model on
    the held-out clients (target subsets) versus the remaining clients as well as
    on the official test split. All metrics are printed in a format similar to
    the unlearning pipeline for easy side-by-side comparison.
    """

    # Setup
    for client_id in excluded_client_ids:
        assert 0 <= client_id < num_clients, (
            f"excluded_client_id must be in [0, {num_clients-1}]"
        )
    target_subset_ids = excluded_client_ids  # naming consistency

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

    excluded_keys = [f"client{i + 1}" for i in excluded_client_ids]
    included_clients_data = {k: v for k, v in clients_data.items() if k not in excluded_keys}

    # Dataset **without** the excluded clients
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
        f"\n[BASELINE] Training on {num_clients - len(excluded_client_ids)} clients (excluding clients {excluded_client_ids})"
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
    # 1) Training data loader (contains only the samples actually seen during optimisation)
    #    Validation samples are excluded to ensure subset-ID accuracy is computed strictly on the training set.
    training_eval_loader = DataLoader(train_dataset, batch_size=batch_size)
    
    # 2) Target subset loader (contains only the excluded clients)
    excluded_clients_data = {key: clients_data[key] for key in excluded_keys}
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
    # Calculate accuracy on the training data (remaining clients)
    train_digit_acc = calculate_overall_digit_classification_accuracy(model, training_eval_loader, device)

    # Subset-ID accuracy • training set only (other subsets)
    _sub_tgt_dummy, train_subset_others_acc = calculate_subset_identification_accuracy_multiple_targets(
        model, training_eval_loader, device, target_subset_ids
    )

    # Subset-ID accuracy • target subset (not part of training set)
    target_subset_acc, _sub_oth_dummy = calculate_subset_identification_accuracy_multiple_targets(
        model, target_subset_loader, device, target_subset_ids
    )

    # Digit accuracy on target subset only
    target_digit_acc = calculate_overall_digit_classification_accuracy(model, target_subset_loader, device)

    # Consolidate values for summary output
    tgt_acc = target_digit_acc
    oth_acc = train_digit_acc
    sub_tgt_acc = target_subset_acc
    sub_oth_acc = train_subset_others_acc

    # For multi-target evaluation, we'll average the test accuracies
    test_digit_tgt_total = 0
    test_digit_oth_total = 0
    
    for target_id in target_subset_ids:
        tgt, oth = calculate_digit_classification_accuracy(model, test_loader, device, target_id)
        test_digit_tgt_total += tgt
        test_digit_oth_total += oth

    test_digit_tgt = test_digit_tgt_total / len(target_subset_ids) if target_subset_ids else 0
    test_digit_oth = test_digit_oth_total / len(target_subset_ids) if target_subset_ids else 0
    
    test_digit_acc = calculate_overall_digit_classification_accuracy(model, test_loader, device)

    # Print summary
    print("[BASELINE] ---------------------------------------------")

    # Per-client accuracies on the excluded (target) clients
    for client_id in excluded_client_ids:
        key = f"client{client_id + 1}"
        client_only_dataset = MultiTaskDataset(full_dataset, {key: clients_data[key]})
        client_only_loader = DataLoader(client_only_dataset, batch_size=batch_size)
        # Compute per-client accuracy (treat the client's ID as the target subset)
        client_tgt_acc, _ = calculate_digit_classification_accuracy(
            model, client_only_loader, device, target_subset_id=client_id
        )
        print(f"Digit accuracy on client{client_id + 1}: {client_tgt_acc:.4f}")
        # Also report per-client subset-ID accuracy (MTL only)
        client_sub_tgt_acc, _ = calculate_subset_identification_accuracy(
            model, client_only_loader, device, target_subset_id=client_id
        )
        print(f"Subset ID accuracy on client{client_id + 1}: {client_sub_tgt_acc:.4f}")

    print(f"Digit accuracy on other subsets: {oth_acc:.4f}")
    print(f"Subset ID accuracy on target subsets: {sub_tgt_acc:.4f}")
    print(f"Subset ID accuracy on other subsets: {sub_oth_acc:.4f}")
    print("    [TEST] Digit accuracy: {:.4f}".format(test_digit_acc))

    # MIA (train-only)
    mia_score = get_membership_attack_prob_train_only(training_eval_loader, target_subset_loader, model)
    print(f"Train-only MIA Accuracy (target subsets vs retain): {mia_score:.2f}%")

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

def learn_baseline_excluding_2_clients(
    **kwargs,
):
    """Wrapper to exclude 2 clients."""
    return learn_baseline_excluding_clients(excluded_client_ids=[0, 1], **kwargs)


def learn_baseline_excluding_3_clients(
    **kwargs,
):
    """Wrapper to exclude 3 clients."""
    return learn_baseline_excluding_clients(excluded_client_ids=[0, 1, 2], **kwargs)


def learn_baseline_excluding_2_clients_ce_only(
    **kwargs,
):
    """Wrapper to exclude 2 clients with CE-only MTL (disentanglement disabled)."""
    kwargs.setdefault("lambda_dis", 0.0)
    return learn_baseline_excluding_clients(excluded_client_ids=[0, 1], **kwargs)


def learn_baseline_excluding_clients_ce_only(
    excluded_client_ids,
    **kwargs,
):
    """General wrapper to train CE-only MTL baseline with selected clients excluded."""
    kwargs.setdefault("lambda_dis", 0.0)
    return learn_baseline_excluding_clients(excluded_client_ids=excluded_client_ids, **kwargs)
