#!/usr/bin/env python3
import random
import torch
from torch.utils.data import DataLoader, Subset

from utils import set_global_seed
from data import generate_subdatasets, MultiTaskDataset
from training import train_mtl_two_heads
from models import MTL_Two_Heads_ResNet

def train_baseline_all_clients(
    dataset_name: str = "CIFAR10",
    setting: str = "non-iid",
    num_clients: int = 10,
    batch_size: int = 128,
    num_epochs: int = 200,
    lambda_1: float = 1.0,
    lambda_2: float = 1.0,
    lambda_dis: float = 0.1,
    lambda_pull: float = 1.0,
    lambda_push: float = 1.0,
    data_root: str = "./data",
    path: str = "baseline_mtl_all_clients.h5",
    model_class=MTL_Two_Heads_ResNet,
    seed: int = 42,
    head_size: str = "medium",
):
    """Train a baseline MTL model on *all* clients."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    set_global_seed(seed)

    clients_data, _, full_dataset = generate_subdatasets(
        dataset_name=dataset_name,
        setting=setting,
        num_clients=num_clients,
        data_root=data_root,
    )

    mtl_dataset = MultiTaskDataset(full_dataset, clients_data)

    dataset_size = len(mtl_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    indices = list(range(dataset_size))
    random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(mtl_dataset, train_indices)
    val_dataset = Subset(mtl_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = model_class(dataset_name=dataset_name, num_clients=num_clients, head_size=head_size)
    model.to(device)
    
    print(f"\n[BASELINE-ALL] Training on all {num_clients} clients")
    
    model, history = train_mtl_two_heads(
        model=model,
        train_loader=train_loader,
        test_loader=val_loader,
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
    print(f"[BASELINE-ALL] Model weights saved to {path}\n")

    return model, history

if __name__ == "__main__":
    train_baseline_all_clients(
        dataset_name="CIFAR10",
        num_clients=10,
        head_size="medium",
        seed=42,
        path="baseline_mtl_all_clients.h5",
        num_epochs=200
    ) 