import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict

from evaluation import (
    calculate_digit_classification_accuracy,
    calculate_subset_identification_accuracy,
    calculate_overall_digit_classification_accuracy,
    calculate_overall_subset_identification_accuracy,
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

    def calc_importance(self, dataloader: DataLoader) -> Dict[str, torch.Tensor]:
        """Compute Fisher-style importance scores for subset identification task."""
        criterion = nn.CrossEntropyLoss()
        importances = self._zero_like_param_dict()
        if len(dataloader.dataset) == 0:
            return importances

        for inputs, _digit_labels, subset_labels in dataloader:
            inputs = inputs.to(self.device)
            subset_labels = subset_labels.to(self.device)

            # forward through the model – we care about subset head here
            _digit_logits, subset_logits, _ = self.model(inputs)
            loss = criterion(subset_logits, subset_labels)

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
) -> nn.Module:
    """Apply Selective Synaptic Dampening (SSD) to forget a target subset.

    This method does NOT perform any further fine-tuning – the weights are
    multiplicatively dampened in a single shot.  The routine mirrors the output
    style of the other unlearning pipelines for consistency.
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

    print("Computing parameter importances on retain/general set …")
    imp_retain = perturber.calc_importance(retain_loader)
    print("Computing parameter importances on forget set …")
    imp_forget = perturber.calc_importance(forget_loader)

    print("Applying synaptic dampening …")
    perturber.apply_dampening(imp_retain, imp_forget)

    # Accuracy evaluation (post-dampening, pre-fine-tune)
    print("\nCalculating accuracies after dampening…")

    # Combine retain and forget datasets for a single pass evaluation
    combined_dataset = [(x, y_dig, y_sub) for loader in (retain_loader, forget_loader) for x, y_dig, y_sub in loader.dataset]

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
    if test_loader is not None:
        test_digit_acc = calculate_overall_digit_classification_accuracy(unlearned_model, test_loader, device)
        test_subset_acc = calculate_overall_subset_identification_accuracy(unlearned_model, test_loader, device)
        print(f"[TEST] Digit accuracy after SSD: {test_digit_acc:.4f}")

    # Optional evaluation (MIA) for feedback (train-only variant)
    mia_score = get_membership_attack_prob_train_only(retain_loader, forget_loader, unlearned_model)
    print(f"Train-only MIA Score on forget set after SSD: {mia_score:.4f}")

    print("--- SSD Unlearning Finished ---")
    return unlearned_model 