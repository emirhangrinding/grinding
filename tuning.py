import optuna
import torch
from torch.utils.data import DataLoader
from typing import Optional, Dict
import io

from utils import set_global_seed, calculate_baseline_delta_score
from ssd import ssd_unlearn_subset
from evaluation import (
    calculate_digit_classification_accuracy,
    calculate_subset_identification_accuracy,
    calculate_overall_digit_classification_accuracy,
)
import copy

def optimise_ssd_hyperparams(
    pretrained_model,
    retain_loader,
    forget_loader,
    test_loader,
    device,
    target_subset_id: Optional[int],
    n_trials: int = 25,
    seed: int = 49,
    calculate_fisher_on: str = "subset",
    num_forgotten_clients: int = 1,
    unlearned_model_name: str = "unlearned_model",
    all_forgotten_loaders: Optional[Dict[int, DataLoader]] = None,
    *,
    kill_output_neuron: bool = False,
    digit_metrics_only: bool = False,
    baseline_variant: Optional[str] = None,  # one of {"mtl", "mtl_ce", "no_mtl"}
    current_client_id: Optional[int] = None,
):
    """Run TPE search to tune SSD hyper-parameters α (exponent) and λ (dampening_constant).
    The optimisation minimises the distance to baseline metrics based on the number of forgotten clients.
    Lower score ⇒ closer to baseline performance.
    """

    # Infer baseline variant if not provided
    inferred_no_mtl = (target_subset_id is None)
    if baseline_variant is None:
        baseline_variant = "no_mtl" if inferred_no_mtl else "mtl"

    is_no_mtl = (baseline_variant == "no_mtl")

    # Determine which client ID we are forgetting this round (needed for per-client baselines)
    if current_client_id is None:
        current_client_id = target_subset_id if target_subset_id is not None else 0

    # Define baseline metrics per round and per client
    # For each round we keep per-client target accuracies and global "other" metrics
    BASELINES = {}
    if baseline_variant == "no_mtl":
        BASELINES = {
            1: {
                'per_client_target_digit_acc': {0: 0.5868},
                'other_digit_acc': 0.9999,
                'test_digit_acc': 0.6206,
            },
            2: {
                'per_client_target_digit_acc': {0: 0.5520, 1: 0.5399},
                'other_digit_acc': 0.9998,
                'test_digit_acc': 0.6002,
            },
            3: {
                'per_client_target_digit_acc': {0: 0.5372, 1: 0.5224, 2: 0.4991},
                'other_digit_acc': 0.9998,
                'test_digit_acc': 0.5774,
            },
        }
    elif baseline_variant == "mtl_ce":
        BASELINES = {
            1: {
                'per_client_target_digit_acc': {0: 0.6414},
                'other_digit_acc': 0.9997,
                'target_subset_acc': 0.0000,
                'other_subset_acc': 0.9994,
                'test_digit_acc': 0.6691,
            },
            2: {
                'per_client_target_digit_acc': {0: 0.6161, 1: 0.6072},
                'other_digit_acc': 0.9998,
                'target_subset_acc': 0.0000,
                'other_subset_acc': 0.9993,
                'test_digit_acc': 0.6570,
            },
            3: {
                'per_client_target_digit_acc': {0: 0.5990, 1: 0.5971, 2: 0.5620},
                'other_digit_acc': 0.9998,
                'target_subset_acc': 0.0000,
                'other_subset_acc': 0.9995,
                'test_digit_acc': 0.6413,
            },
        }
    else:  # "mtl"
        BASELINES = {
            1: {
                'per_client_target_digit_acc': {0: 0.6435},
                'other_digit_acc': 0.9998,
                'target_subset_acc': 0.0000,
                'other_subset_acc': 0.9995,
                'test_digit_acc': 0.6908,
            },
            2: {
                'per_client_target_digit_acc': {0: 0.6241, 1: 0.6151},
                'other_digit_acc': 0.9998,
                'target_subset_acc': 0.0000,
                'other_subset_acc': 0.9966,
                'test_digit_acc': 0.6686,
            },
            3: {
                'per_client_target_digit_acc': {0: 0.6085, 1: 0.6035, 2: 0.5776},
                'other_digit_acc': 0.9998,
                'target_subset_acc': 0.0000,
                'other_subset_acc': 0.9996,
                'test_digit_acc': 0.6495,
            },
        }

    def _make_client_round_baseline(round_idx: int, client_id: int) -> Dict[str, float]:
        round_info = BASELINES[round_idx]
        client_map = round_info.get('per_client_target_digit_acc', {})
        if client_id not in client_map:
            # Fallback: if outside provided range, use the last available client's metric
            # or the first if empty
            if client_map:
                last_key = sorted(client_map.keys())[-1]
                target_acc = client_map[last_key]
            else:
                target_acc = 0.0
        else:
            target_acc = client_map[client_id]

        # Build baseline dict for scoring
        baseline = {
            'target_digit_acc': target_acc,
            'other_digit_acc': round_info.get('other_digit_acc', None),
            'test_digit_acc': round_info.get('test_digit_acc', None),
        }
        if not is_no_mtl:
            baseline.update({
                'target_subset_acc': round_info.get('target_subset_acc', 0.0),
                'other_subset_acc': round_info.get('other_subset_acc', 0.0),
            })
        # Remove None values
        baseline = {k: v for k, v in baseline.items() if v is not None}
        return baseline

    # Helper filter when optimizing only on digit metrics
    def _filter_digit_only(metrics_dict):
        if not digit_metrics_only:
            return metrics_dict
        allowed = {'target_digit_acc', 'other_digit_acc', 'test_digit_acc'}
        return {k: v for k, v in metrics_dict.items() if k in allowed}

    # Establish the current stage (round) for baseline selection
    current_round = min(max(num_forgotten_clients, 1), 3)

    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=10, n_ei_candidates=24)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def _objective(trial):
        alpha = trial.suggest_float("alpha", 0.1, 100.0, log=True)
        lam = trial.suggest_float("lambda", 0.01, 10.0, log=True)

        unlearned_model, initial_ssd_metrics = ssd_unlearn_subset(
            pretrained_model,
            retain_loader,
            forget_loader,
            target_subset_id,
            device,
            exponent=alpha,
            dampening_constant=lam,
            test_loader=test_loader,
            calculate_fisher_on=calculate_fisher_on,
            kill_output_neuron=kill_output_neuron,
            silent=True,
        )

        model_state = io.BytesIO()
        torch.save(unlearned_model.state_dict(), model_state)
        trial.set_user_attr("model_state", model_state.getvalue())

        # --- Compute total deviation from baseline digit accuracy across ALL forgotten clients ---
        total_delta_score = 0.0

        # Print: previously forgotten clients first (if any), then newly forgotten, then others/test
        if num_forgotten_clients > 1 and all_forgotten_loaders:
            # Ensure proper masking for MTL during evaluation
            if not is_no_mtl and hasattr(unlearned_model, "kill_output_neuron"):
                unlearned_model.kill_output_neuron = True
                if hasattr(unlearned_model, "killed_subset_ids"):
                    mask_ids = set(all_forgotten_loaders.keys())
                    if target_subset_id is not None:
                        mask_ids.add(int(target_subset_id))
                    unlearned_model.killed_subset_ids = mask_ids
                elif hasattr(unlearned_model, "killed_subset_id") and (target_subset_id is not None):
                    unlearned_model.killed_subset_id = int(target_subset_id)

            for client_id, loader in all_forgotten_loaders.items():
                # Evaluate metrics per previously forgotten client
                if is_no_mtl:
                    tdig = calculate_overall_digit_classification_accuracy(unlearned_model, loader, device)
                    tsub = 0.0
                else:
                    tdig, _ = calculate_digit_classification_accuracy(
                        unlearned_model, loader, device, target_subset_id=client_id
                    )
                    tsub, _ = calculate_subset_identification_accuracy(
                        unlearned_model, loader, device, target_subset_id=client_id
                    )

                # Logging in requested format
                print(f"digit accuracy on client {client_id} after SSD (previously forgotten): {tdig:.4f}")
                if not is_no_mtl:
                    print(f"subset id accuracy on client {client_id} after SSD (previously forgotten): {tsub:.4f}")

                # Sum absolute deviation from baseline target-digit accuracy (per-client)
                client_baseline = _make_client_round_baseline(current_round, client_id)
                baseline_tdig = client_baseline.get('target_digit_acc', 0.0)
                total_delta_score += abs(tdig - baseline_tdig)

                # Record per-client metrics on the trial for later inspection
                trial.set_user_attr(f"client_{client_id}_digit_acc", tdig)
                if not is_no_mtl:
                    trial.set_user_attr(f"client_{client_id}_subset_acc", tsub)

        # Newly forgotten client (current)
        print(f"digit accuracy on client {current_client_id} after SSD (newly forgotten): {initial_ssd_metrics['target_digit_acc']:.4f}")
        print(f"digit accuracy on other subsets after SSD: {initial_ssd_metrics['other_digit_acc']:.4f}")
        if not is_no_mtl:
            print(f"subset id accuracy on client {current_client_id} after SSD (newly forgotten): {initial_ssd_metrics.get('target_subset_acc', 0.0):.4f}")
            print(f"subset id accuracy on other subsets after SSD: {initial_ssd_metrics.get('other_subset_acc', 0.0):.4f}")
        if initial_ssd_metrics.get('test_digit_acc') is not None:
            print(f"[TEST] Digit accuracy after SSD: {initial_ssd_metrics['test_digit_acc']:.4f}")

        # Add deviation for the newly forgotten client vs baseline
        current_baseline = _make_client_round_baseline(current_round, current_client_id)
        total_delta_score += abs(initial_ssd_metrics['target_digit_acc'] - current_baseline.get('target_digit_acc', 0.0))

        # Log main trial metrics
        trial.set_user_attr("target_digit_acc", initial_ssd_metrics['target_digit_acc'])
        trial.set_user_attr("other_digit_acc", initial_ssd_metrics['other_digit_acc'])
        trial.set_user_attr("test_digit_acc", initial_ssd_metrics['test_digit_acc'])
        trial.set_user_attr("mia_score", initial_ssd_metrics.get('mia_score', -1))
        if not is_no_mtl:
            trial.set_user_attr("target_subset_acc", initial_ssd_metrics['target_subset_acc'])
            trial.set_user_attr("other_subset_acc", initial_ssd_metrics['other_subset_acc'])

        return total_delta_score

    study.optimize(_objective, n_trials=n_trials)

    print(f"\n--- SSD Hyperparameter Optimisation Finished ---")
    print(f"Best α: {study.best_params['alpha']:.6f}, Best λ: {study.best_params['lambda']:.6f}")
    print(f"Best combined delta score: {study.best_value:.6f}")

    best_trial = study.best_trial
    # Print best-trial metrics in the requested format (re-evaluate using the saved best model)
    # Reconstruct the best model for consistent reporting
    best_state = torch.load(io.BytesIO(best_trial.user_attrs["model_state"]), map_location=device)
    best_model = copy.deepcopy(pretrained_model).to(device)
    best_model.load_state_dict(best_state)
    best_model.eval()

    # Mask subset neurons if applicable
    if not is_no_mtl and hasattr(best_model, "kill_output_neuron"):
        best_model.kill_output_neuron = True
        if hasattr(best_model, "killed_subset_ids") and all_forgotten_loaders:
            mask_ids = set(all_forgotten_loaders.keys())
            if target_subset_id is not None:
                mask_ids.add(int(target_subset_id))
            best_model.killed_subset_ids = mask_ids
        elif hasattr(best_model, "killed_subset_id") and (target_subset_id is not None):
            best_model.killed_subset_id = int(target_subset_id)

    # Print previously forgotten clients
    if num_forgotten_clients > 1 and all_forgotten_loaders:
        for client_id, loader in all_forgotten_loaders.items():
            if is_no_mtl:
                tdig = calculate_overall_digit_classification_accuracy(best_model, loader, device)
                tsub = 0.0
            else:
                tdig, _ = calculate_digit_classification_accuracy(best_model, loader, device, target_subset_id=client_id)
                tsub, _ = calculate_subset_identification_accuracy(best_model, loader, device, target_subset_id=client_id)
            print(f"digit accuracy on client {client_id} after SSD (previously forgotten): {tdig:.4f}")
            if not is_no_mtl:
                print(f"subset id accuracy on client {client_id} after SSD (previously forgotten): {tsub:.4f}")

    # Newly forgotten client metrics and 'other subsets' on best model
    if is_no_mtl:
        # No-MTL: treat as overall metrics on loaders
        current_tdig = calculate_overall_digit_classification_accuracy(best_model, forget_loader, device)
        other_dig = calculate_overall_digit_classification_accuracy(best_model, retain_loader, device)
        print(f"digit accuracy on client {current_client_id} after SSD (newly forgotten): {current_tdig:.4f}")
        print(f"digit accuracy on other subsets after SSD: {other_dig:.4f}")
    else:
        # Build a combined loader to compute other-subsets metrics correctly
        combined_list = []
        for loader in (retain_loader, forget_loader):
            for item in loader.dataset:
                if len(item) == 3:
                    x, y_dig, y_sub = item
                else:
                    x, y_dig = item
                    y_sub = 0
                combined_list.append((x, y_dig, y_sub))

        class _TempDataset(torch.utils.data.Dataset):
            def __init__(self, data_list):
                self.data = data_list
            def __len__(self):
                return len(self.data)
            def __getitem__(self, idx):
                return self.data[idx]

        temp_loader = DataLoader(_TempDataset(combined_list), batch_size=retain_loader.batch_size, shuffle=False)

        cur_tdig, oth_dig = calculate_digit_classification_accuracy(best_model, temp_loader, device, current_client_id)
        cur_tsub, oth_sub = calculate_subset_identification_accuracy(best_model, temp_loader, device, current_client_id)
        print(f"digit accuracy on client {current_client_id} after SSD (newly forgotten): {cur_tdig:.4f}")
        print(f"digit accuracy on other subsets after SSD: {oth_dig:.4f}")
        print(f"subset id accuracy on client {current_client_id} after SSD (newly forgotten): {cur_tsub:.4f}")
        print(f"subset id accuracy on other subsets after SSD: {oth_sub:.4f}")

    # Test accuracy on best model
    if test_loader is not None:
        best_test_acc = calculate_overall_digit_classification_accuracy(best_model, test_loader, device)
        print(f"[TEST] Digit accuracy after SSD: {best_test_acc:.4f}")

    output_filename = f"{unlearned_model_name}.h5"
    best_model_state_bytes = best_trial.user_attrs["model_state"]
    with open(output_filename, "wb") as f:
        f.write(best_model_state_bytes)
    print(f"\n✓ Best unlearned model saved to: {output_filename}")

    # --- Evaluate and print metrics for previously forgotten clients using the best model ---
    # (Legacy block replaced by unified best-model reporting above)

    return study
