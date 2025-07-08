#!/usr/bin/env python3
"""
Complete script to run SSD hyperparameter optimization.
This script loads a pretrained model and optimizes α (exponent) and λ (dampening_constant) 
to match baseline metrics including the 75.44% MIA score.
"""

import random
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST, CIFAR10

from utils import set_global_seed, SEED_DEFAULT
from data import generate_subdatasets, MultiTaskDataset, transform_mnist, transform_test_cifar, create_subset_data_loaders
from models import MTL_Two_Heads_ResNet
from tuning import optimise_ssd_hyperparams

def run_ssd_hyperparameter_optimization(
    model_path: str,
    dataset_name: str = "CIFAR10",
    target_subset_id: int = 0,
    num_clients: int = 10,
    batch_size: int = 64,
    n_trials: int = 25,
    data_root: str = "./data",
    head_size: str = "big",
    seed: int = SEED_DEFAULT
):
    """
    Complete pipeline to run SSD hyperparameter optimization.
    
    Args:
        model_path: Path to pretrained model weights
        dataset_name: "CIFAR10" or "MNIST" 
        target_subset_id: Which client/subset to "forget" (0-based)
        num_clients: Number of clients/subsets
        batch_size: Batch size for data loaders
        n_trials: Number of optimization trials
        data_root: Root directory for datasets
        head_size: Size of classification heads ("big" or "small")
        seed: Random seed for reproducibility
    """
    
    print("=== SSD Hyperparameter Optimization ===")
    print(f"Model path: {model_path}")
    print(f"Dataset: {dataset_name}")
    print(f"Target subset ID: {target_subset_id}")
    print(f"Number of clients: {num_clients}")
    print(f"Number of trials: {n_trials}")
    print(f"Seed: {seed}")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set global seeds for reproducibility
    set_global_seed(seed)
    
    # === 1. Generate training data ===
    print("\n=== Setting up training data ===")
    clients_data, clients_labels, full_dataset = generate_subdatasets(
        dataset_name=dataset_name,
        setting="non-iid",
        num_clients=num_clients,
        data_root=data_root
    )
    
    # Create multi-task dataset
    mtl_dataset = MultiTaskDataset(full_dataset, clients_data)
    
    # Train/validation split (80/20) - we only need training part for optimization
    dataset_size = len(mtl_dataset)
    train_size = int(0.8 * dataset_size)
    indices = list(range(dataset_size))
    random.shuffle(indices)
    train_indices = indices[:train_size]
    
    train_dataset = Subset(mtl_dataset, train_indices)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Training samples: {len(train_dataset)}")
    
    # === 2. Create retain and forget loaders ===
    print(f"\n=== Creating retain/forget split for target subset {target_subset_id} ===")
    retain_loader, forget_loader = create_subset_data_loaders(train_loader, target_subset_id)
    
    print(f"Retain samples: {len(retain_loader.dataset)}")
    print(f"Forget samples: {len(forget_loader.dataset)}")
    
    # === 3. Setup test data loader (following exact pattern from codebase) ===
    print("\n=== Setting up test data ===")
    
    # Load official test split
    if dataset_name == "MNIST":
        test_base = MNIST(root=data_root, train=False, download=True, transform=transform_mnist)
    else:  # CIFAR10
        test_base = CIFAR10(root=data_root, train=False, download=True, transform=transform_test_cifar)
    
    # Create class-based distribution for test data (same as training)
    test_class_indices = {i: [] for i in range(10)}
    for idx, (_, label) in enumerate(test_base):
        test_class_indices[label].append(idx)
    
    # Distribute test samples to clients using same mapping as training
    test_clients_data = {k: [] for k in clients_data.keys()}
    for class_label, idxs in test_class_indices.items():
        # Find which clients have this class in training data
        client_ids = [cid for cid in clients_data.keys() 
                     if any(full_dataset[i][1] == class_label for i in clients_data[cid])]
        if not client_ids:
            client_ids = list(clients_data.keys())
        
        # Round-robin assignment to clients
        for i, idx in enumerate(idxs):
            client = client_ids[i % len(client_ids)]
            test_clients_data[client].append(idx)
    
    # Create test MultiTaskDataset and DataLoader
    test_mtl_dataset = MultiTaskDataset(test_base, test_clients_data)
    test_loader = DataLoader(test_mtl_dataset, batch_size=batch_size)
    
    print(f"Test samples: {len(test_mtl_dataset)}")
    
    # === 4. Load pretrained model ===
    print(f"\n=== Loading pretrained model from {model_path} ===")
    model = MTL_Two_Heads_ResNet(
        dataset_name=dataset_name, 
        num_clients=num_clients, 
        head_size=head_size
    )
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print("✓ Model loaded successfully")
    except FileNotFoundError:
        print(f"❌ Error: Model file not found at {model_path}")
        print("Please ensure you have a trained model at the specified path.")
        return None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    # === 5. Run hyperparameter optimization ===
    print(f"\n=== Starting SSD hyperparameter optimization with {n_trials} trials ===")
    print("Optimizing to match baseline metrics:")
    print("  • target_digit_acc: 0.9056")
    print("  • other_digit_acc: 0.9998") 
    print("  • target_subset_acc: 0.0000")
    print("  • other_subset_acc: 0.9974")
    print("  • test_digit_acc: 0.9130")
    print("  • mia_score: 75.44%")
    print()
    
    study = optimise_ssd_hyperparams(
        pretrained_model=model,
        retain_loader=retain_loader,
        forget_loader=forget_loader,
        test_loader=test_loader,
        device=device,
        target_subset_id=target_subset_id,
        n_trials=n_trials,
        seed=seed
    )
    
    # === 6. Summary ===
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE!")
    print("="*60)
    print(f"Best α (exponent): {study.best_params['alpha']:.6f}")
    print(f"Best λ (lambda): {study.best_params['lambda']:.6f}")
    print(f"Best delta score: {study.best_value:.6f}")
    print("\nYou can now use these hyperparameters in your SSD unlearning!")
    
    return study

if __name__ == "__main__":
    # Configuration - EDIT THESE VALUES FOR YOUR SETUP
    MODEL_PATH = "path_to_your_pretrained_model.pth"  # ⚠️ CHANGE THIS!
    DATASET_NAME = "CIFAR10"                          # "CIFAR10" or "MNIST"
    TARGET_SUBSET_ID = 0                              # Which client to "forget" (0-based)
    NUM_CLIENTS = 10                                  # Number of clients/subsets
    BATCH_SIZE = 64                                   # Batch size
    N_TRIALS = 25                                     # Number of optimization trials
    DATA_ROOT = "./data"                              # Data directory
    HEAD_SIZE = "big"                                 # "big" or "small"
    SEED = 42                                         # Random seed
    
    # Validate configuration
    if MODEL_PATH == "path_to_your_pretrained_model.pth":
        print("❌ Please edit MODEL_PATH in the script to point to your actual pretrained model!")
        print("Example: MODEL_PATH = './models/cifar10_pretrained.pth'")
        exit(1)
    
    # Run optimization
    study = run_ssd_hyperparameter_optimization(
        model_path=MODEL_PATH,
        dataset_name=DATASET_NAME,
        target_subset_id=TARGET_SUBSET_ID,
        num_clients=NUM_CLIENTS,
        batch_size=BATCH_SIZE,
        n_trials=N_TRIALS,
        data_root=DATA_ROOT,
        head_size=HEAD_SIZE,
        seed=SEED
    ) 