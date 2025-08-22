import random
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from collections import defaultdict, Counter

from utils import dirichlet_partition

# Dataset-specific transforms

# Training transform for CIFAR10 – replaced with simpler resize→crop pipeline as requested
transform_train_cifar = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                        std=(0.2023, 0.1994, 0.2010)),
])

# Deterministic transform for CIFAR10 test/validation data
transform_test_cifar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                        std=(0.2023, 0.1994, 0.2010)),
])

# Default transform for MNIST (keeps previous behaviour)
transform_mnist = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def generate_subdatasets(
    dataset_name='MNIST',
    setting='iid',
    num_clients=10,
    data_root='./data'
):
    # Dataset-specific transforms
    if dataset_name == 'MNIST':
        dataset = MNIST(root=data_root, train=True, download=True, transform=transform_mnist)
        num_classes = 10
    elif dataset_name == 'CIFAR10':
        dataset = CIFAR10(root=data_root, train=True, download=True, transform=transform_train_cifar)
        num_classes = 10
    elif dataset_name == 'CIFAR100':
        dataset = CIFAR100(root=data_root, train=True, download=True, transform=transform_train_cifar)
        num_classes = 100
    else:
        raise AssertionError("Dataset must be 'MNIST', 'CIFAR10', or 'CIFAR100'")
    assert setting in ['iid', 'non-iid', 'extreme-non-iid'], "Invalid setting"
    total_samples = len(dataset)
    samples_per_client = total_samples // num_clients

    # Organize indices by class
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    clients_data = {}
    clients_labels = {}

    if setting == 'iid':
        # IID: Each client gets a random subset with balanced class distribution
        all_indices = list(range(total_samples))
        random.shuffle(all_indices)

        for client_id in range(num_clients):
            start_idx = client_id * samples_per_client
            end_idx = start_idx + samples_per_client
            client_indices = all_indices[start_idx:end_idx]
            clients_data[f"client{client_id+1}"] = client_indices
            labels = [dataset[i][1] for i in client_indices]
            clients_labels[f"client{client_id+1}"] = dict(Counter(labels))

    elif setting == 'non-iid':
        # Non-IID Dirichlet split
        dirichlet_alpha = 0.6      # You may expose as an argument if you want
        client_split_indices = dirichlet_partition(
            dataset, num_clients=num_clients, alpha=dirichlet_alpha, num_classes=num_classes, seed=42
        )
        for client_id, client_idxs in enumerate(client_split_indices):
            clients_data[f"client{client_id+1}"] = client_idxs
            labels = [dataset[i][1] for i in client_idxs]
            clients_labels[f"client{client_id+1}"] = dict(Counter(labels))

    elif setting == 'extreme-non-iid':
        # Extreme Non-IID: Each client gets samples from exactly one class
        assert num_clients >= num_classes, "Number of clients must be at least equal to number of classes"

        # Assign each class to clients evenly
        clients_per_class = num_clients // num_classes
        remaining_clients = num_clients % num_classes

        client_idx = 0
        for class_label in range(num_classes):
            # Calculate how many clients should get this class
            num_assigned_clients = clients_per_class + (1 if class_label < remaining_clients else 0)

            # Split the indices for this class among the assigned clients
            class_samples = class_indices[class_label]
            samples_per_assigned_client = len(class_samples) // num_assigned_clients

            for i in range(num_assigned_clients):
                start_idx = i * samples_per_assigned_client
                end_idx = start_idx + samples_per_assigned_client if i < num_assigned_clients - 1 else len(class_samples)
                client_samples = class_samples[start_idx:end_idx]
                clients_data[f"client{client_idx+1}"] = client_samples
                clients_labels[f"client{client_idx+1}"] = {class_label: len(client_samples)}
                client_idx += 1

    return clients_data, clients_labels, dataset

class MultiTaskDataset(Dataset):
    def __init__(self, dataset, client_indices):
        self.dataset = dataset
        self.client_mapping = {}

        # Map each data index to its corresponding client ID
        for client_id, indices in client_indices.items():
            client_num = int(client_id.replace('client', '')) - 1  # Convert 'client1' to 0, etc.
            for idx in indices:
                self.client_mapping[idx] = client_num

        # Keep only indices that have been assigned to clients
        self.valid_indices = list(self.client_mapping.keys())

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        dataset_idx = self.valid_indices[idx]
        x, main_label = self.dataset[dataset_idx]
        client_label = self.client_mapping[dataset_idx]
        return x, main_label, client_label

def create_subset_data_loaders(train_loader, target_subset_id):
    """
    Create retain and forget loaders based on the target subset ID.
    """
    forget_indices = []
    retain_indices = []

    for idx in range(len(train_loader.dataset)):
        inputs, digit_labels, subset_labels = train_loader.dataset[idx]
        if subset_labels == target_subset_id:
            forget_indices.append(idx)
        else:
            retain_indices.append(idx)

    forget_dataset = torch.utils.data.Subset(train_loader.dataset, forget_indices)
    retain_dataset = torch.utils.data.Subset(train_loader.dataset, retain_indices)

    forget_loader = torch.utils.data.DataLoader(forget_dataset, batch_size=train_loader.batch_size, shuffle=False)
    retain_loader = torch.utils.data.DataLoader(retain_dataset, batch_size=train_loader.batch_size, shuffle=False)

    return retain_loader, forget_loader 