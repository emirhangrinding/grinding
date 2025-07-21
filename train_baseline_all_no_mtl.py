import random
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision.datasets import MNIST, CIFAR10

from utils import set_global_seed, SEED_DEFAULT
from data import generate_subdatasets, transform_mnist, transform_test_cifar
from training import train_single_head, train_single_head_with_eval
from models import StandardResNet
from evaluation import (
    calculate_overall_digit_classification_accuracy,
    get_membership_attack_prob_train_only,
    calculate_digit_classification_accuracy
)


def learn_baseline_all_clients(
    dataset_name: str = "CIFAR10",
    setting: str = "non-iid",
    num_clients: int = 10,
    target_client_id: int = 0,
    batch_size: int = 128,
    num_epochs: int = 200,
    data_root: str = "./data",
    path: str = "baseline_all_clients_model.h5",
    model_class=StandardResNet,
    seed: int = SEED_DEFAULT,
    target_test_accuracy: float = None,
):
    """Train a baseline model on *all* clients.

    After training, the routine evaluates the model on the target client 
    (which will be 'forgotten' later) versus the remaining clients, as well as
    on the official test split. All metrics are printed for comparison.
    """

    # Setup
    assert 0 <= target_client_id < num_clients, (
        f"target_client_id must be in [0, {num_clients-1}]"
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

    all_indices = []
    for k, v in clients_data.items():
        all_indices.extend(v)

    # Dataset with all the clients
    full_train_dataset = Subset(full_dataset, all_indices)

    # Train/val split (same 80/20 rule) - not used for training, but for consistency
    dataset_size = len(full_train_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    indices = list(range(dataset_size))
    random.shuffle(indices)
    train_indices = indices[:train_size]
    # val_indices = indices[train_size:] # Not used

    train_dataset = Subset(full_train_dataset, train_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    if dataset_name == "MNIST":
        test_base = MNIST(root=data_root, train=False, download=True, transform=transform_mnist)
    else:
        test_base = CIFAR10(root=data_root, train=False, download=True, transform=transform_test_cifar)
    test_loader = DataLoader(test_base, batch_size=batch_size)

    # Model & training
    model = model_class(dataset_name=dataset_name)

    # Create target subset loader for evaluation during training
    target_key = f"client{target_client_id + 1}"
    target_subset_dataset = Subset(full_dataset, clients_data[target_key])
    target_subset_loader = DataLoader(target_subset_dataset, batch_size=batch_size)
    
    # Create training evaluation loader (for the included clients)
    training_eval_loader = DataLoader(train_dataset, batch_size=batch_size)

    print(
        f"\n[BASELINE] Training on all {num_clients} clients"
    )
    model, history = train_single_head_with_eval(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        training_eval_loader=training_eval_loader,
        target_subset_loader=target_subset_loader,
        device=device,
        num_epochs=num_epochs,
        dataset_name=dataset_name,
        target_test_accuracy=target_test_accuracy,
    )

    torch.save(model.state_dict(), path)
    print(f"[BASELINE] Model weights saved to {path}\n")

    # Metrics
    # The evaluation loaders were already created for training evaluation.
    # We will use the training set that was actually used for training, not the full dataset
    # to be consistent with the epoch-by-epoch evaluation
    
    train_digit_acc = calculate_overall_digit_classification_accuracy(model, training_eval_loader, device)

    # Digit accuracy on target subset only
    target_digit_acc = calculate_overall_digit_classification_accuracy(model, target_subset_loader, device)
    
    test_digit_acc = calculate_overall_digit_classification_accuracy(model, test_loader, device)

    # Print summary
    print("[BASELINE] ---------------------------------------------")
    print(f"Digit accuracy on target subset: {target_digit_acc:.4f}")
    print(f"Digit accuracy on training data (80% subset): {train_digit_acc:.4f}")
    print(f"Digit accuracy on test set: {test_digit_acc:.4f}")

    # MIA (train-only) - Use the training eval loader instead of full train loader
    mia_score = get_membership_attack_prob_train_only(training_eval_loader, target_subset_loader, model)
    print(f"Train-only MIA Accuracy (target subset vs training data): {mia_score:.2f}%")

    print("[BASELINE] ---------------------------------------------\n")

    return model, history, {
        "train_digit_tgt": target_digit_acc,
        "train_digit_training": train_digit_acc,  # Changed from "train_digit_all" to be more accurate
        "test_digit_acc": test_digit_acc,
        "mia_score": mia_score,
    } 


if __name__ == "__main__":
    # Run the training with default parameters
    model, history, metrics = learn_baseline_all_clients()
    print(f"\nTraining completed. Final metrics: {metrics}") 