import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict

from evaluation import (
    calculate_digit_classification_accuracy,
    calculate_subset_identification_accuracy,
    calculate_overall_digit_classification_accuracy,
    get_membership_attack_prob_train_only
)

class ParameterPerturber:
    """Utility class implementing Selective Synaptic Dampening (SSD).

    The class is intentionally lightweight and task-specific: it focuses on the
    subset-identification head that we are interested in unlearning.  It follows
    the same spirit as the original SSD paper – parameters that are more
    important for the forget set than the retain/general set are multiplicatively
    dampened (never increased).
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        *,
        lower_bound: float = 1.0,
        exponent: float = 1.0,
        dampening_constant: float = 0.5,
        selection_weighting: float = 1.0,
    ) -> None:
        self.model = model
        self.device = device
        self.lower_bound = lower_bound
        self.exponent = exponent
        self.dampening_constant = dampening_constant
        self.selection_weighting = selection_weighting

    def _zero_like_param_dict(self) -> Dict[str, torch.Tensor]:
        """Return a dict with the same keys as model.named_parameters() but zeros."""
        return {name: torch.zeros_like(p, device=self.device) for name, p in self.model.named_parameters()}

    def calc_importance(self, dataloader: DataLoader, calculate_fisher_on: str = "subset") -> Dict[str, torch.Tensor]:
        """Compute Fisher-style importance scores for the specified task."""
        if calculate_fisher_on not in ["digit", "subset"]:
            raise ValueError(f"calculate_fisher_on must be 'digit' or 'subset', got {calculate_fisher_on}")
            
        criterion = nn.CrossEntropyLoss()
        importances = self._zero_like_param_dict()
        if len(dataloader.dataset) == 0:
            return importances

        for batch in dataloader:
            # Handle both MTL (3 values) and no-MTL (2 values) data formats
            if len(batch) == 3:
                inputs, digit_labels, subset_labels = batch
                is_mtl = True
            else:  # len(batch) == 2
                inputs, digit_labels = batch
                subset_labels = None
                is_mtl = False
                
            inputs = inputs.to(self.device)
            
            # Choose the appropriate labels based on task
            if calculate_fisher_on == "digit":
                labels = digit_labels.to(self.device)
            else:  # calculate_fisher_on == "subset"
                if not is_mtl:
                    # For no-MTL case, fallback to digit task since subset task doesn't exist
                    #print(f"Warning: subset task not available in no-MTL case, using digit task instead")
                    labels = digit_labels.to(self.device)
                    calculate_fisher_on_current = "digit"  # Override for this iteration
                else:
                    labels = subset_labels.to(self.device)
                    calculate_fisher_on_current = "subset"

            # Forward through the model - handle both single-head and multi-head outputs
            model_outputs = self.model(inputs)
            if isinstance(model_outputs, tuple):
                # Multi-head model (MTL case)
                digit_logits, subset_logits, _ = model_outputs
            else:
                # Single-head model (no-MTL case)
                digit_logits = model_outputs
                subset_logits = None
            
            # Choose the appropriate logits based on task
            if calculate_fisher_on == "digit" or (not is_mtl and calculate_fisher_on == "subset"):
                loss = criterion(digit_logits, labels)
            else:  # calculate_fisher_on == "subset" and is_mtl
                loss = criterion(subset_logits, labels)

            # accumulate squared gradients
            self.model.zero_grad()
            loss.backward()
            for (name, param) in self.model.named_parameters():
                if param.grad is not None:
                    importances[name] += param.grad.data.pow(2)

        # average over batches
        for name in importances:
            importances[name] /= float(len(dataloader))
        return importances

    def apply_dampening(self, original_imp: Dict[str, torch.Tensor], forget_imp: Dict[str, torch.Tensor]) -> None:
        """Dampen parameters whose importance is dominated by the forget set."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                oimp = original_imp.get(name)
                fimp = forget_imp.get(name)
                if oimp is None or fimp is None:
                    continue

                # SSD selection criterion
                mask = fimp > (oimp * self.selection_weighting)
                if not mask.any():
                    continue

                # Dampening factor (lambda in the paper)
                weight = ((oimp * self.dampening_constant) / (fimp + 1e-9)).pow(self.exponent)
                update = weight[mask]
                # ensure we never *increase* a weight (>1)
                update[update > self.lower_bound] = self.lower_bound

                param[mask] = param[mask] * update

def ssd_unlearn_subset(
    pretrained_model: nn.Module,
    retain_loader: DataLoader,
    forget_loader: DataLoader,
    target_subset_id: int,
    device: torch.device,
    *,
    lower_bound: float = 1.0,
    exponent: float = 1.0,
    dampening_constant: float = 0.5,
    selection_weighting: float = 1.0,
    test_loader: DataLoader = None,
    calculate_fisher_on: str = "subset",
) -> tuple:
    """Apply Selective Synaptic Dampening (SSD) to forget a target subset.

    This method does NOT perform any further fine-tuning – the weights are
    multiplicatively dampened in a single shot.  The routine mirrors the output
    style of the other unlearning pipelines for consistency.
    
    Returns:
        tuple: (unlearned_model, metrics_dict) where metrics_dict contains the calculated accuracies
    """

    print("\n--- Starting Selective Synaptic Dampening (SSD) Unlearning ---")
    unlearned_model = copy.deepcopy(pretrained_model).to(device)

    perturber = ParameterPerturber(
        unlearned_model,
        device,
        lower_bound=lower_bound,
        exponent=exponent,
        dampening_constant=dampening_constant,
        selection_weighting=selection_weighting,
    )

    print(f"Computing parameter importances on retain/general set for {calculate_fisher_on} task …")
    imp_retain = perturber.calc_importance(retain_loader, calculate_fisher_on)
    print(f"Computing parameter importances on forget set for {calculate_fisher_on} task …")
    imp_forget = perturber.calc_importance(forget_loader, calculate_fisher_on)

    print("Applying synaptic dampening …")
    perturber.apply_dampening(imp_retain, imp_forget)

    # Accuracy evaluation (post-dampening, pre-fine-tune)
    print("\nCalculating accuracies after dampening…")

    # Combine retain and forget datasets for a single pass evaluation
    combined_dataset = []
    for loader in (retain_loader, forget_loader):
        for item in loader.dataset:
            if len(item) == 3:
                # MultiTaskDataset format
                x, y_dig, y_sub = item
                combined_dataset.append((x, y_dig, y_sub))
            else:
                # Standard dataset format - use dummy subset label
                x, y_dig = item
                combined_dataset.append((x, y_dig, 0))  # Use 0 as dummy subset label

    class _TempDataset(torch.utils.data.Dataset):
        def __init__(self, data_list):
            self.data = data_list
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]

    temp_loader = DataLoader(_TempDataset(combined_dataset), batch_size=retain_loader.batch_size, shuffle=False)

    tgt_acc, oth_acc = calculate_digit_classification_accuracy(unlearned_model, temp_loader, device, target_subset_id)
    sub_tgt_acc, sub_oth_acc = calculate_subset_identification_accuracy(unlearned_model, temp_loader, device, target_subset_id)

    print(f"Digit accuracy on target subset after SSD: {tgt_acc:.4f}")
    print(f"Digit accuracy on other subsets after SSD: {oth_acc:.4f}")
    print(f"Subset ID accuracy on target subset after SSD: {sub_tgt_acc:.4f}")
    print(f"Subset ID accuracy on other subsets after SSD: {sub_oth_acc:.4f}")

    # Test set evaluation
    test_digit_acc = None
    if test_loader is not None:
        test_digit_acc = calculate_overall_digit_classification_accuracy(unlearned_model, test_loader, device)
        print(f"[TEST] Digit accuracy after SSD: {test_digit_acc:.4f}")

    # Optional evaluation (MIA) for feedback (train-only variant)
    mia_score = get_membership_attack_prob_train_only(retain_loader, forget_loader, unlearned_model)
    print(f"Train-only MIA Score on forget set after SSD: {mia_score:.4f}")

    print("--- SSD Unlearning Finished ---")
    
    # Return both model and metrics
    metrics = {
        'target_digit_acc': tgt_acc,
        'other_digit_acc': oth_acc,
        'target_subset_acc': sub_tgt_acc,
        'other_subset_acc': sub_oth_acc,
        'test_digit_acc': test_digit_acc,
        'mia_score': mia_score
    }
    
    return unlearned_model, metrics 