import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10

from utils import set_global_seed, intra_y1_y2_disentanglement_loss, SEED_DEFAULT
from data import (
    generate_subdatasets, MultiTaskDataset, transform_mnist, transform_test_cifar
)
from models import MTL_Two_Heads_ResNet

def train_mtl_two_heads(model, train_loader, test_loader, device,
                        num_epochs=10, lambda_1=1.0, lambda_2=1.0, lambda_dis=0.1, 
                        lambda_pull=1.0, lambda_push=1.0, dataset_name='CIFAR10'):
    """
    Train the multi-task model with two classification heads.
    - For digit classification: Use both train and validation sets
    - For subset identification: Allow overfitting by using only training set
    """
    model = model.to(device)
    
    # Dataset-specific optimizer configuration (matching machine-unlearning-disentangle)
    if dataset_name == 'MNIST':
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = None
    else:  # CIFAR10
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

    # Loss functions - both using cross-entropy
    digit_criterion = nn.CrossEntropyLoss()
    subset_criterion = nn.CrossEntropyLoss()

    history = {
        'train_digit_loss': [], 'train_subset_loss': [], 'train_loss_dis': [], 'train_total_loss': [],
        'train_digit_acc': [], 'train_subset_acc': [],
        'test_digit_loss': [], 'test_digit_acc': [],
    }

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_digit_loss, train_subset_loss = 0.0, 0.0
        train_loss_dis = 0.0
        train_digit_correct, train_subset_correct = 0, 0
        train_samples = 0

        for inputs, digit_labels, subset_labels in train_loader:
            inputs = inputs.to(device)
            digit_labels = digit_labels.to(device)
            subset_labels = subset_labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            digit_logits, subset_logits, embeddings = model(inputs, return_features=True)

            # Calculate losses
            digit_loss = digit_criterion(digit_logits, digit_labels)
            subset_loss = subset_criterion(subset_logits, subset_labels)
            loss_dis = intra_y1_y2_disentanglement_loss(embeddings, digit_labels, subset_labels, 
                                                        lambda_pull, lambda_push)

            # ----- Conditional loss schedule (only digit loss for first 20 epochs) -----
            if epoch < 20:  # epochs indexed from 0, so epoch 0–19 => epochs 1–20
                total_loss = lambda_1 * digit_loss
            else:
                total_loss = lambda_1 * digit_loss + lambda_2 * subset_loss + lambda_dis * loss_dis

            # Backward and optimize
            total_loss.backward()
            optimizer.step()

            # Track metrics
            train_digit_loss += digit_loss.item() * inputs.size(0)
            train_subset_loss += subset_loss.item() * inputs.size(0)
            if isinstance(loss_dis, torch.Tensor):
                train_loss_dis += loss_dis.item() * inputs.size(0)
            else:
                train_loss_dis += loss_dis * inputs.size(0)

            # Calculate accuracies
            _, digit_preds = torch.max(digit_logits, 1)
            _, subset_preds = torch.max(subset_logits, 1)
            train_digit_correct += (digit_preds == digit_labels).sum().item()
            train_subset_correct += (subset_preds == subset_labels).sum().item()
            train_samples += inputs.size(0)

        # Step the scheduler if it exists
        if scheduler is not None:
            scheduler.step()

        # Compute training metrics (average per batch for consistency with machine-unlearning-disentangle)
        num_batches = len(train_loader)
        epoch_train_digit_loss = train_digit_loss / num_batches
        epoch_train_subset_loss = train_subset_loss / num_batches
        epoch_loss_dis = train_loss_dis / num_batches
        if epoch < 20:
            epoch_train_total_loss = lambda_1 * epoch_train_digit_loss  # matches optimisation objective
        else:
            epoch_train_total_loss = lambda_1 * epoch_train_digit_loss + lambda_2 * epoch_train_subset_loss + lambda_dis * epoch_loss_dis
        epoch_train_digit_acc = train_digit_correct / train_samples
        epoch_train_subset_acc = train_subset_correct / train_samples

        # Evaluation on official test split (only digit classification)
        model.eval()
        test_digit_loss = 0.0
        test_digit_correct = 0
        test_samples = 0

        with torch.no_grad():
            for batch in test_loader:
                # Handle both 2-value and 3-value batches
                if len(batch) == 2:
                    inputs, digit_labels = batch
                else:  # len(batch) == 3
                    inputs, digit_labels, _ = batch  # ignore subset_labels
                    
                inputs = inputs.to(device)
                digit_labels = digit_labels.to(device)

                # Forward pass
                digit_logits, _, _ = model(inputs)

                # Calculate loss
                digit_loss = digit_criterion(digit_logits, digit_labels)

                # Track metrics
                test_digit_loss += digit_loss.item() * inputs.size(0)

                # Calculate accuracy
                _, digit_preds = torch.max(digit_logits, 1)
                test_digit_correct += (digit_preds == digit_labels).sum().item()
                test_samples += inputs.size(0)

        # Compute validation metrics
        epoch_val_digit_loss = test_digit_loss / test_samples
        epoch_val_digit_acc = test_digit_correct / test_samples

        # Update history
        history['train_digit_loss'].append(epoch_train_digit_loss)
        history['train_subset_loss'].append(epoch_train_subset_loss)
        history['train_total_loss'].append(epoch_train_total_loss)
        history['train_digit_acc'].append(epoch_train_digit_acc)
        history['train_subset_acc'].append(epoch_train_subset_acc)
        history['test_digit_loss'].append(epoch_val_digit_loss)
        history['test_digit_acc'].append(epoch_val_digit_acc)
        history['train_loss_dis'].append(epoch_loss_dis)

        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Main Loss (Train): {epoch_train_digit_loss:.4f}, Main Acc (Train): {epoch_train_digit_acc:.4f}, '
              f'Main Acc (Test) : {epoch_val_digit_acc:.4f}, '
              f'Subset Loss (Train): {epoch_train_subset_loss:.4f}, Subset Acc (Train): {epoch_train_subset_acc:.4f}')

        # Save model checkpoints at specific epochs
        if (epoch + 1) in {20, 50, 100}:
            checkpoint_path = f"model_{epoch + 1}.h5"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    return model, history


def train_single_head(
    model,
    train_loader,
    test_loader,
    device,
    num_epochs,
    dataset_name,
    target_test_accuracy=None,
):
    """Train a single-head model for standard classification."""

    model.to(device)

    # Dataset-specific optimizer configuration (optimized for baseline training)
    if dataset_name == 'MNIST':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        # Simple step scheduler for MNIST - reduces LR every 10 epochs
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    else:  # CIFAR10
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        # CosineAnnealingLR is more suitable for baseline CIFAR10 training
        # T_max should match the expected number of epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    criterion = torch.nn.CrossEntropyLoss()
    history = {"train_loss": [], "test_acc": []}

    print(f"[INFO] Using scheduler: {type(scheduler).__name__} for {dataset_name}")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        history["train_loss"].append(epoch_loss)

        # Step the scheduler if it exists
        if scheduler is not None:
            scheduler.step()

        # Evaluate on test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_acc = 100 * correct / total
        history["test_acc"].append(test_acc)

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Test Accuracy: {test_acc:.2f}%"
        )

        if target_test_accuracy is not None and test_acc >= target_test_accuracy:
            print(
                f"Target test accuracy of {target_test_accuracy}% reached. Stopping training."
            )
            break

    return model, history


def train_single_head_with_eval(
    model,
    train_loader,
    test_loader,
    training_eval_loader,
    target_subset_loader,
    device,
    num_epochs,
    dataset_name,
    target_test_accuracy=None,
):
    """Train a single-head model for standard classification with epoch-by-epoch evaluation."""
    from evaluation import calculate_overall_digit_classification_accuracy

    model.to(device)

    # Dataset-specific optimizer configuration (optimized for baseline training)
    if dataset_name == 'MNIST':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        # Simple step scheduler for MNIST - reduces LR every 10 epochs
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    else:  # CIFAR10
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        # CosineAnnealingLR is more suitable for baseline CIFAR10 training
        # T_max should match the expected number of epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    criterion = torch.nn.CrossEntropyLoss()
    history = {"train_loss": [], "test_acc": [], "train_acc": [], "target_acc": []}

    print(f"[INFO] Using scheduler: {type(scheduler).__name__} for {dataset_name}")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        history["train_loss"].append(epoch_loss)

        # Step the scheduler if it exists
        if scheduler is not None:
            scheduler.step()

        # Evaluate on test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_acc = 100 * correct / total
        history["test_acc"].append(test_acc)
        
        # Evaluate on training set (included clients)
        train_acc = calculate_overall_digit_classification_accuracy(model, training_eval_loader, device) * 100
        history["train_acc"].append(train_acc)
        
        # Evaluate on target subset (excluded client)
        target_acc = calculate_overall_digit_classification_accuracy(model, target_subset_loader, device) * 100
        history["target_acc"].append(target_acc)

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, "
            f"Test Acc: {test_acc:.2f}%, Train Acc: {train_acc:.2f}%, Target Acc: {target_acc:.2f}%"
        )

        if target_test_accuracy is not None and test_acc >= target_test_accuracy:
            print(
                f"Target test accuracy of {target_test_accuracy}% reached. Stopping training."
            )
            break

    return model, history

def learn(dataset_name='MNIST', setting='non-iid', num_clients=10,
          batch_size=64, num_epochs=10, lambda_1=1.0, lambda_2=1.0, lambda_dis=0.1,
          lambda_pull=1.0, lambda_push=1.0, data_root='./data',
          path="model.h5", model_class=MTL_Two_Heads_ResNet, seed: int = SEED_DEFAULT,
          head_size: str = 'big'):
    """
    Run a Multi-Task Learning experiment with two classification heads,
    with optional unlearning.
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set global seeds for reproducibility
    set_global_seed(seed)

    # Generate subdatasets
    clients_data, clients_labels, full_dataset = generate_subdatasets(
        dataset_name=dataset_name,
        setting=setting,
        num_clients=num_clients,
        data_root=data_root
    )

    # Create multi-task dataset
    mtl_dataset = MultiTaskDataset(full_dataset, clients_data)

    # Use the full training set (no internal validation)
    train_loader = DataLoader(mtl_dataset, batch_size=batch_size, shuffle=True)

    # Prepare official test split (digit classification only)
    if dataset_name == 'MNIST':
        test_base = MNIST(root=data_root, train=False, download=True, transform=transform_mnist)
    else:
        test_base = CIFAR10(root=data_root, train=False, download=True, transform=transform_test_cifar)
    test_loader = DataLoader(test_base, batch_size=batch_size)

    # Create model using the specified model class
    model = model_class(dataset_name=dataset_name, num_clients=num_clients, head_size=head_size)

    # Print summary of dataset
    print(f"Dataset: {dataset_name}, Setting: {setting}, Clients: {num_clients}")
    print(f"Train samples: {len(mtl_dataset)}, Test samples: {len(test_base)}")
    print(f"Batch size: {batch_size}")
    print(f"Loss weights - λ1: {lambda_1}, λ2: {lambda_2}, λ_dis: {lambda_dis}, λ_pull: {lambda_pull}, λ_push: {lambda_push}")
    print(f"Head size setting: {head_size}")
    print(f"Using cross-entropy loss for both tasks")
    print(f"Allowing subset identification head to overfit (no validation)")

    # Train model
    model, history = train_mtl_two_heads(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        num_epochs=num_epochs,
        lambda_1=lambda_1,
        lambda_2=lambda_2,
        lambda_dis=lambda_dis,
        lambda_pull=lambda_pull,
        lambda_push=lambda_push,
        dataset_name=dataset_name
    )

    # Save weights after initial training
    torch.save(model.state_dict(), path)
    print(f"Model weights saved to {path}")

    return model, history, clients_labels 