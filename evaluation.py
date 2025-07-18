import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score

def calculate_digit_classification_accuracy(model, data_loader, device, target_subset_id):
    """
    Calculate digit classification accuracy for the target subset and other subsets.
    """
    model.eval()
    target_correct = 0
    other_correct = 0
    target_samples = 0
    other_samples = 0

    with torch.no_grad():
        for inputs, digit_labels, subset_labels in data_loader:
            inputs, digit_labels, subset_labels = \
                inputs.to(device), digit_labels.to(device), subset_labels.to(device)
            digit_logits, _, _ = model(inputs)
            _, digit_preds = torch.max(digit_logits, 1)

            target_mask = (subset_labels == target_subset_id)
            other_mask = ~target_mask

            target_correct += (digit_preds[target_mask] == digit_labels[target_mask]).sum().item()
            other_correct += (digit_preds[other_mask] == digit_labels[other_mask]).sum().item()

            target_samples += target_mask.sum().item()
            other_samples += other_mask.sum().item()

    target_accuracy = target_correct / target_samples if target_samples > 0 else 0.0
    other_accuracy = other_correct / other_samples if other_samples > 0 else 0.0

    return target_accuracy, other_accuracy

def calculate_subset_identification_accuracy(model, data_loader, device, target_subset_id):
    """
    Calculate subset identification accuracy for the target subset and other subsets.
    """
    model.eval()
    target_correct = 0
    other_correct = 0
    target_samples = 0
    other_samples = 0

    with torch.no_grad():
        for inputs, digit_labels, subset_labels in data_loader:
            inputs, digit_labels, subset_labels = \
                inputs.to(device), digit_labels.to(device), subset_labels.to(device)
            _, subset_logits, _ = model(inputs)
            _, subset_preds = torch.max(subset_logits, 1)

            target_mask = (subset_labels == target_subset_id)
            other_mask = ~target_mask

            target_correct += (subset_preds[target_mask] == subset_labels[target_mask]).sum().item()
            other_correct += (subset_preds[other_mask] == subset_labels[other_mask]).sum().item()

            target_samples += target_mask.sum().item()
            other_samples += other_mask.sum().item()

    target_accuracy = target_correct / target_samples if target_samples > 0 else 0.0
    other_accuracy = other_correct / other_samples if other_samples > 0 else 0.0

    return target_accuracy, other_accuracy

def calculate_overall_digit_classification_accuracy(model, data_loader, device):
    """Calculate digit classification accuracy across the entire loader (all subsets combined)."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in data_loader:
            # Handle both 2-value and 3-value batch formats
            if len(batch) == 2:
                inputs, digit_labels = batch
                inputs, digit_labels = inputs.to(device), digit_labels.to(device)
            else:
                inputs, digit_labels, _ = batch
                inputs, digit_labels = inputs.to(device), digit_labels.to(device)
            
            # Handle both single-head and multi-head model outputs
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                digit_logits = outputs[0]  # Multi-head model
            else:
                digit_logits = outputs  # Single-head model
            
            _, digit_preds = torch.max(digit_logits, 1)
            correct += (digit_preds == digit_labels).sum().item()
            total += digit_labels.size(0)
    return correct / total if total > 0 else 0.0

def _get_model_outputs(loader, model, device):
    """Return concatenated probability vectors for digit and subset heads."""
    model.eval()
    outputs = []
    with torch.no_grad():
        for batch in loader:
            # Handle both 2-value and 3-value unpacking from data loaders
            if len(batch) == 3:
                inputs, _digit_labels, _subset_labels = batch
            else:  # len(batch) == 2
                inputs, _digit_labels = batch
            
            inputs = inputs.to(device)
            model_outputs = model(inputs)
            
            # Handle both MTL models (3 outputs) and standard models (1 output)
            if isinstance(model_outputs, tuple) and len(model_outputs) == 3:
                # MTL model with two heads
                digit_logits, subset_logits, _ = model_outputs
                digit_probs = torch.softmax(digit_logits, dim=1)
                subset_probs = torch.softmax(subset_logits, dim=1)
                probs = torch.cat((digit_probs, subset_probs), dim=1)
            else:
                # Standard model with single head
                digit_logits = model_outputs
                digit_probs = torch.softmax(digit_logits, dim=1)
                # For compatibility with MTL evaluation, we'll create dummy subset probs
                # This is only used for MIA attacks which focus on the output distribution
                dummy_subset_probs = torch.zeros(digit_probs.size(0), 1, device=digit_probs.device)
                probs = torch.cat((digit_probs, dummy_subset_probs), dim=1)
            
            outputs.append(probs.cpu().numpy())

    if not outputs:
        return np.array([])
    return np.concatenate(outputs, axis=0)

def get_membership_attack_prob_train_only(retain_loader, forget_loader, model):
    """Compute train-only Membership Inference Attack (MIA) *accuracy*.

    A LogisticRegression attacker is trained (with 5-fold CV) to distinguish
    retain (label 0) from forget (label 1) samples using the model's combined
    soft-max outputs from both heads.

    The function now returns the *overall classification accuracy* of this
    attacker expressed as a percentage (0-100).  A return value of −1 is kept
    for the corner-case where either loader lacks data.
    """
    device = next(model.parameters()).device

    # Gather model outputs
    retain_outputs = _get_model_outputs(retain_loader, model, device)
    forget_outputs = _get_model_outputs(forget_loader, model, device)

    if len(retain_outputs) == 0 or len(forget_outputs) == 0:
        print("Warning: Not enough data for train-only MIA attack. Returning -1.")
        return -1.0

    # Build dataset for the attacker
    X = np.concatenate([retain_outputs, forget_outputs])
    y = np.concatenate([
        np.zeros(len(retain_outputs), dtype=int),  # retain samples labelled 0
        np.ones(len(forget_outputs), dtype=int),   # forget samples labelled 1
    ])

    # Train the attacker via cross-validated predictions
    clf = LogisticRegression(
        class_weight="balanced",
        solver="lbfgs",
        multi_class="multinomial",
        max_iter=1000,
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(clf, X, y, cv=skf)

    # Compute overall accuracy and convert to percentage
    accuracy = accuracy_score(y, y_pred) * 100.0
    return accuracy 