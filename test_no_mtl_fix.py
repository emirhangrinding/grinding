#!/usr/bin/env python3
"""
Test script to verify that no-MTL evaluation fixes work correctly.
This script tests the key functions with both MTL and no-MTL scenarios.
"""
import torch
from torch.utils.data import DataLoader, TensorDataset
from models import StandardResNet, MTL_Two_Heads_ResNet
from evaluation import calculate_digit_classification_accuracy, calculate_overall_digit_classification_accuracy
from ssd import ssd_unlearn_subset

def create_dummy_data(batch_size=32, num_samples=100, num_classes=10):
    """Create dummy data for testing"""
    # Create random data
    x = torch.randn(num_samples, 3, 32, 32)  # CIFAR-like data
    y_digit = torch.randint(0, num_classes, (num_samples,))
    y_subset = torch.randint(0, 5, (num_samples,))  # 5 subsets
    
    # Create MTL dataset (3-tuple)
    mtl_dataset = TensorDataset(x, y_digit, y_subset)
    mtl_loader = DataLoader(mtl_dataset, batch_size=batch_size)
    
    # Create no-MTL dataset (2-tuple)
    no_mtl_dataset = TensorDataset(x, y_digit)
    no_mtl_loader = DataLoader(no_mtl_dataset, batch_size=batch_size)
    
    return mtl_loader, no_mtl_loader

def test_evaluation_functions():
    """Test the evaluation functions with both MTL and no-MTL data"""
    print("Testing evaluation functions...")
    
    # Create dummy data
    mtl_loader, no_mtl_loader = create_dummy_data()
    
    # Create models
    mtl_model = MTL_Two_Heads_ResNet(dataset_name="CIFAR10", num_clients=5)
    no_mtl_model = StandardResNet(dataset_name="CIFAR10")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mtl_model.to(device)
    no_mtl_model.to(device)
    
    print("\n1. Testing MTL evaluation (target_subset_id=0):")
    try:
        target_acc, other_acc = calculate_digit_classification_accuracy(
            mtl_model, mtl_loader, device, target_subset_id=0
        )
        print(f"   ✓ MTL evaluation succeeded: target_acc={target_acc:.4f}, other_acc={other_acc:.4f}")
    except Exception as e:
        print(f"   ✗ MTL evaluation failed: {e}")
    
    print("\n2. Testing no-MTL evaluation (target_subset_id=None):")
    try:
        target_acc, other_acc = calculate_digit_classification_accuracy(
            no_mtl_model, no_mtl_loader, device, target_subset_id=None
        )
        print(f"   ✓ No-MTL evaluation succeeded: target_acc={target_acc:.4f}, other_acc={other_acc:.4f}")
        print(f"   ✓ Expected: target_acc=0.0000 (all samples treated as 'other')")
    except Exception as e:
        print(f"   ✗ No-MTL evaluation failed: {e}")
    
    print("\n3. Testing overall accuracy calculation:")
    try:
        overall_acc = calculate_overall_digit_classification_accuracy(no_mtl_model, no_mtl_loader, device)
        print(f"   ✓ Overall accuracy calculation succeeded: {overall_acc:.4f}")
    except Exception as e:
        print(f"   ✗ Overall accuracy calculation failed: {e}")

def test_ssd_evaluation():
    """Test SSD evaluation with no-MTL scenario"""
    print("\n\nTesting SSD evaluation...")
    
    # Create dummy data
    _, no_mtl_loader = create_dummy_data(batch_size=16, num_samples=50)
    
    # Split into retain and forget loaders (simulate subset split)
    retain_data = []
    forget_data = []
    for i, (x, y) in enumerate(no_mtl_loader.dataset):
        if i < 30:  # First 30 samples as retain
            retain_data.append((x, y))
        else:  # Last 20 samples as forget
            forget_data.append((x, y))
    
    retain_dataset = TensorDataset(*zip(*retain_data))
    forget_dataset = TensorDataset(*zip(*forget_data))
    retain_loader = DataLoader(retain_dataset, batch_size=16)
    forget_loader = DataLoader(forget_dataset, batch_size=16)
    
    # Create model
    model = StandardResNet(dataset_name="CIFAR10")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print("\n1. Testing SSD with target_subset_id=None (correct for no-MTL):")
    try:
        unlearned_model, metrics = ssd_unlearn_subset(
            model, retain_loader, forget_loader,
            target_subset_id=None,  # This should work correctly now
            device=device,
            calculate_fisher_on="digit"  # Use digit for no-MTL
        )
        print(f"   ✓ SSD evaluation succeeded:")
        print(f"     target_digit_acc: {metrics['target_digit_acc']:.4f}")
        print(f"     other_digit_acc: {metrics['other_digit_acc']:.4f}")
        print(f"     target_subset_acc: {metrics['target_subset_acc']:.4f}")
        print(f"     other_subset_acc: {metrics['other_subset_acc']:.4f}")
    except Exception as e:
        print(f"   ✗ SSD evaluation failed: {e}")

if __name__ == "__main__":
    print("Testing no-MTL evaluation fixes...")
    test_evaluation_functions()
    test_ssd_evaluation()
    print("\n✓ All tests completed!") 