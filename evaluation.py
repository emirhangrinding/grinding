import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score
from typing import Optional, Tuple, List, Dict

def calculate_digit_classification_accuracy(model, data_loader, device, target_subset_id: Optional[int]) -> Tuple[float, float]:
    """
    Calculate digit classification accuracy for the target subset and other subsets.
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader containing the evaluation data
        device: Device to run evaluation on
        target_subset_id: ID of the target subset to evaluate, or None for no-MTL models
        
    Returns:
        Tuple of (target_accuracy, other_accuracy)
        For no-MTL models, target_accuracy will be 0.0 and other_accuracy will be the overall accuracy
    """
    model.eval()
    target_correct = 0
    other_correct = 0
    target_samples = 0
    other_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            # Handle both MTL (3 values) and no-MTL (2 values) data formats
            if len(batch) == 3:
                inputs, digit_labels, subset_labels = batch
                is_mtl = True
            else:  # len(batch) == 2
                inputs, digit_labels = batch
                subset_labels = torch.zeros_like(digit_labels)  # Use dummy subset labels
                is_mtl = False
                
            inputs, digit_labels, subset_labels = \
                inputs.to(device), digit_labels.to(device), subset_labels.to(device)
            
            # Handle both single-head and multi-head model outputs
            model_outputs = model(inputs)
            if isinstance(model_outputs, tuple):
                # Multi-head model (MTL case)
                digit_logits, _, _ = model_outputs
            else:
                # Single-head model (no-MTL case)
                digit_logits = model_outputs
                
            _, digit_preds = torch.max(digit_logits, 1)

            # Handle the case where target_subset_id is None (no-MTL case)
            if target_subset_id is None:
                # For no-MTL models, treat all samples as "other" samples
                target_mask = torch.zeros_like(subset_labels, dtype=torch.bool)
                other_mask = torch.ones_like(subset_labels, dtype=torch.bool)
            else:
                target_mask = (subset_labels == target_subset_id)
                other_mask = ~target_mask

                # Ensure masks are tensors, not scalars
                if not isinstance(target_mask, torch.Tensor):
                    print(f"Warning: target_mask is not a tensor: {type(target_mask)}, subset_labels shape: {subset_labels.shape}, target_subset_id: {target_subset_id}")
                    target_mask = torch.tensor([target_mask], device=subset_labels.device).expand_as(subset_labels)
                    other_mask = ~target_mask
                elif target_mask.numel() == 1 and subset_labels.numel() > 1:
                    # Handle case where comparison results in scalar for vector input
                    target_mask = target_mask.expand_as(subset_labels)
                    other_mask = ~target_mask

            target_correct += (digit_preds[target_mask] == digit_labels[target_mask]).sum().item()
            other_correct += (digit_preds[other_mask] == digit_labels[other_mask]).sum().item()

            target_samples += target_mask.sum().item()
            other_samples += other_mask.sum().item()

    target_accuracy = target_correct / target_samples if target_samples > 0 else 0.0
    other_accuracy = other_correct / other_samples if other_samples > 0 else 0.0

    return target_accuracy, other_accuracy

def calculate_subset_identification_accuracy(model, data_loader, device, target_subset_id: Optional[int]) -> Tuple[float, float]:
    """
    Calculate subset identification accuracy for the target subset and other subsets.
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader containing the evaluation data
        device: Device to run evaluation on
        target_subset_id: ID of the target subset to evaluate, or None for no-MTL models
        
    Returns:
        Tuple of (target_accuracy, other_accuracy)
        For no-MTL models, target_accuracy will be 0.0 and other_accuracy will be the overall accuracy
    """
    model.eval()
    target_correct = 0
    other_correct = 0
    target_samples = 0
    other_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            # Handle both MTL (3 values) and no-MTL (2 values) data formats
            if len(batch) == 3:
                inputs, digit_labels, subset_labels = batch
                is_mtl = True
            else:  # len(batch) == 2
                inputs, digit_labels = batch
                subset_labels = torch.zeros_like(digit_labels)  # Use dummy subset labels
                is_mtl = False
                
            inputs, digit_labels, subset_labels = \
                inputs.to(device), digit_labels.to(device), subset_labels.to(device)
            
            # Handle both single-head and multi-head model outputs
            model_outputs = model(inputs)
            if isinstance(model_outputs, tuple):
                # Multi-head model (MTL case)
                _, subset_logits, _ = model_outputs
                _, subset_preds = torch.max(subset_logits, 1)
            else:
                # Single-head model (no-MTL case) - subset identification not meaningful
                # Return dummy predictions (all zeros) since there's no subset head
                subset_preds = torch.zeros_like(subset_labels)

            # Handle the case where target_subset_id is None (no-MTL case)
            if target_subset_id is None:
                # For no-MTL models, treat all samples as "other" samples
                target_mask = torch.zeros_like(subset_labels, dtype=torch.bool)
                other_mask = torch.ones_like(subset_labels, dtype=torch.bool)
            else:
                target_mask = (subset_labels == target_subset_id)
                other_mask = ~target_mask

                # Ensure masks are tensors, not scalars
                if not isinstance(target_mask, torch.Tensor):
                    print(f"Warning: target_mask is not a tensor: {type(target_mask)}, subset_labels shape: {subset_labels.shape}, target_subset_id: {target_subset_id}")
                    target_mask = torch.tensor([target_mask], device=subset_labels.device).expand_as(subset_labels)
                    other_mask = ~target_mask
                elif target_mask.numel() == 1 and subset_labels.numel() > 1:
                    # Handle case where comparison results in scalar for vector input
                    target_mask = target_mask.expand_as(subset_labels)
                    other_mask = ~target_mask

            target_correct += (subset_preds[target_mask] == subset_labels[target_mask]).sum().item()
            other_correct += (subset_preds[other_mask] == subset_labels[other_mask]).sum().item()

            target_samples += target_mask.sum().item()
            other_samples += other_mask.sum().item()

    target_accuracy = target_correct / target_samples if target_samples > 0 else 0.0
    other_accuracy = other_correct / other_samples if other_samples > 0 else 0.0

    return target_accuracy, other_accuracy

def calculate_subset_identification_accuracy_multiple_targets(model, data_loader, device, target_subset_ids: Optional[List[int]]) -> Tuple[float, float]:
    """
    Calculate subset identification accuracy for multiple target subsets and other subsets.
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader containing the evaluation data
        device: Device to run evaluation on
        target_subset_ids: IDs of the target subsets to evaluate, or None for no-MTL models
        
    Returns:
        Tuple of (target_accuracy, other_accuracy)
        For no-MTL models, target_accuracy will be 0.0 and other_accuracy will be the overall accuracy
    """
    model.eval()
    target_correct = 0
    other_correct = 0
    target_samples = 0
    other_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            if len(batch) == 3:
                inputs, _, subset_labels = batch
            else:
                inputs, _ = batch
                subset_labels = torch.zeros_like(_)
            
            inputs, subset_labels = inputs.to(device), subset_labels.to(device)
            
            model_outputs = model(inputs)
            if isinstance(model_outputs, tuple):
                _, subset_logits, _ = model_outputs
                _, subset_preds = torch.max(subset_logits, 1)
            else:
                subset_preds = torch.zeros_like(subset_labels)

            if target_subset_ids is None:
                target_mask = torch.zeros_like(subset_labels, dtype=torch.bool)
                other_mask = torch.ones_like(subset_labels, dtype=torch.bool)
            else:
                target_mask = torch.zeros_like(subset_labels, dtype=torch.bool)
                for target_id in target_subset_ids:
                    target_mask |= (subset_labels == target_id)
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
                if isinstance(model_outputs, tuple):
                    digit_logits = model_outputs[0]
                else:
                    digit_logits = model_outputs
                digit_probs = torch.softmax(digit_logits, dim=1)
                # The dummy subset probs were confusing the MIA attacker for no-MTL models.
                # The attacker now only receives the digit probabilities.
                probs = digit_probs
            
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

def calculate_metrics_for_clients(
    model,
    is_mtl: bool,
    data_loader,
    device,
    client_id: int,
    description: str,
) -> Dict[str, float]:
    """Helper function to calculate and return a dictionary of metrics for a given client."""
    metrics = {}
    
    # --- Digit Classification Accuracy ---
    target_digit_acc, other_digit_acc = calculate_digit_classification_accuracy(
        model, data_loader, device, target_subset_id=client_id if is_mtl else None
    )
    if is_mtl:
        metrics[f"Digit acc on client {client_id} ({description})"] = target_digit_acc
    else:
        # For no-MTL, the 'other_digit_acc' is the relevant one as it's the overall accuracy
        metrics[f"Digit acc on client {client_id} ({description})"] = other_digit_acc

    # --- Subset Identification Accuracy (only for MTL) ---
    if is_mtl:
        target_subset_acc, other_subset_acc = calculate_subset_identification_accuracy(
            model, data_loader, device, target_subset_id=client_id
        )
        metrics[f"Subset ID acc on client {client_id} ({description})"] = target_subset_acc
        
    return metrics

def evaluate_and_print_metrics(
    model,
    is_mtl: bool,
    retain_loader,
    test_loader,
    device,
    forgotten_client_loaders: Dict[int, torch.utils.data.DataLoader],
    current_forget_client_id: Optional[int] = None,
    *,
    ssd_print_style: bool = False,
):
    """
    Evaluates the model's performance and prints a comprehensive set of metrics,
    including detailed stats for each forgotten client.
    
    Args:
        model: The model to evaluate.
        is_mtl: Boolean indicating if the model is a Multi-Task Learning model.
        retain_loader: DataLoader for the data retained by the model.
        test_loader: DataLoader for the test set.
        device: The device to run evaluation on.
        forgotten_client_loaders: A dictionary mapping forgotten client IDs to their DataLoaders.
        current_forget_client_id: The ID of the client that was just forgotten in the current step.
    """
    model.eval()
    all_metrics = {}

    # If SSD print style requested, print in the requested order and wording,
    # while still computing a comprehensive metrics dict for return.
    if ssd_print_style:
        # Order: previously forgotten clients, then the newly forgotten client
        ordered_keys = list(forgotten_client_loaders.keys())
        if current_forget_client_id is not None and current_forget_client_id in forgotten_client_loaders:
            previously_forgotten = [cid for cid in ordered_keys if cid != current_forget_client_id]
            ordered_ids = previously_forgotten + [current_forget_client_id]
        else:
            ordered_ids = ordered_keys

        # Print metrics for forgotten clients in order
        for cid in ordered_ids:
            loader = forgotten_client_loaders[cid]
            desc = (
                "newly forgotten" if cid == current_forget_client_id else "previously forgotten"
            )

            # Digit accuracy on the specific client
            target_digit_acc, _ = calculate_digit_classification_accuracy(
                model, loader, device, target_subset_id=cid if is_mtl else None
            )
            print(
                f"digit accuracy on client {cid} after SSD ({desc}): {target_digit_acc:.4f}"
            )
            all_metrics[f"Digit acc on client {cid} ({desc})"] = target_digit_acc

            # Subset identification accuracy on the specific client (MTL only)
            if is_mtl:
                target_subset_acc, _ = calculate_subset_identification_accuracy(
                    model, loader, device, target_subset_id=cid
                )
                print(
                    f"subset id accuracy on client {cid} after SSD ({desc}): {target_subset_acc:.4f}"
                )
                all_metrics[f"Subset ID acc on client {cid} ({desc})"] = target_subset_acc

        # Accuracies on other subsets (retain set)
        dummy_id = -1
        _, retain_digit_acc = calculate_digit_classification_accuracy(
            model, retain_loader, device, target_subset_id=dummy_id
        )
        print(f"digit accuracy on other subsets after SSD: {retain_digit_acc:.4f}")
        all_metrics["Digit accuracy on other subsets"] = retain_digit_acc

        if is_mtl:
            _, retain_subset_acc = calculate_subset_identification_accuracy(
                model, retain_loader, device, target_subset_id=dummy_id
            )
            print(f"subset id accuracy on other subsets after SSD: {retain_subset_acc:.4f}")
            all_metrics["Subset ID accuracy on other subsets"] = retain_subset_acc

        # Test set accuracy
        test_acc = calculate_overall_digit_classification_accuracy(model, test_loader, device)
        print(f"[TEST] Digit accuracy after SSD: {test_acc:.4f}")
        all_metrics["Test set accuracy"] = test_acc

        # MIA score if applicable
        if (
            current_forget_client_id is not None
            and current_forget_client_id in forgotten_client_loaders
        ):
            current_forget_loader = forgotten_client_loaders[current_forget_client_id]
            mia_score = get_membership_attack_prob_train_only(
                retain_loader, current_forget_loader, model
            )
            all_metrics["MIA Score (%)"] = mia_score

        return all_metrics

    # --- Default printing style (legacy) ---
    # Metrics on Forgotten Sets
    for client_id, loader in forgotten_client_loaders.items():
        description = (
            "newly forgotten" if client_id == current_forget_client_id else "previously forgotten"
        )
        client_metrics = calculate_metrics_for_clients(
            model, is_mtl, loader, device, client_id, description
        )
        all_metrics.update(client_metrics)

    # Metrics on Retain Set (other subsets)
    dummy_id = -1
    _, retain_digit_acc = calculate_digit_classification_accuracy(
        model, retain_loader, device, target_subset_id=dummy_id
    )
    all_metrics["Digit accuracy on other subsets"] = retain_digit_acc

    if is_mtl:
        _, retain_subset_acc = calculate_subset_identification_accuracy(
            model, retain_loader, device, target_subset_id=dummy_id
        )
        all_metrics["Subset ID accuracy on other subsets"] = retain_subset_acc

    # Metrics on Test Set
    test_acc = calculate_overall_digit_classification_accuracy(model, test_loader, device)
    all_metrics["Test set accuracy"] = test_acc

    # MIA (retain vs current forget) when applicable
    if current_forget_client_id is not None and current_forget_client_id in forgotten_client_loaders:
        current_forget_loader = forgotten_client_loaders[current_forget_client_id]
        mia_score = get_membership_attack_prob_train_only(retain_loader, current_forget_loader, model)
        all_metrics["MIA Score (%)"] = mia_score

    # Print all metrics in a structured way
    print("\n--- Evaluation Metrics ---")
    for key, value in all_metrics.items():
        print(f"{key}: {value:.4f}")
    print("--------------------------\n")

    return all_metrics
