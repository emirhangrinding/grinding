import optuna
import torch
from torch.utils.data import DataLoader
from typing import Optional, Dict
import io

from utils import set_global_seed, calculate_baseline_delta_score
from ssd import ssd_unlearn_subset

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
                'per_client_target_digit_acc': {0: 0.8983},
                'other_digit_acc': 0.9999,
                'test_digit_acc': 0.9061,
            },
            2: {
                'per_client_target_digit_acc': {0: 0.8951, 1: 0.8784},
                'other_digit_acc': 1.0000,
                'test_digit_acc': 0.8931,
            },
            3: {
                'per_client_target_digit_acc': {0: 0.8757, 1: 0.8745, 2: 0.8912},
                'other_digit_acc': 0.9999,
                'test_digit_acc': 0.8919,
            },
        }
    elif baseline_variant == "mtl_ce":
        BASELINES = {
            1: {
                'per_client_target_digit_acc': {0: 0.8975},
                'other_digit_acc': 0.9995,
                'target_subset_acc': 0.0000,
                'other_subset_acc': 0.9909,
                'test_digit_acc': 0.9042,
            },
            2: {
                'per_client_target_digit_acc': {0: 0.8967, 1: 0.8843},
                'other_digit_acc': 0.9995,
                'target_subset_acc': 0.0000,
                'other_subset_acc': 0.9940,
                'test_digit_acc': 0.8986,
            },
            3: {
                'per_client_target_digit_acc': {0: 0.8761, 1: 0.8765, 2: 0.8983},
                'other_digit_acc': 0.9981,
                'target_subset_acc': 0.0000,
                'other_subset_acc': 0.9735,
                'test_digit_acc': 0.8869,
            },
        }
    else:  # "mtl"
        BASELINES = {
            1: {
                'per_client_target_digit_acc': {0: 0.9037},
                'other_digit_acc': 0.9998,
                'target_subset_acc': 0.0000,
                'other_subset_acc': 0.9974,
                'test_digit_acc': 0.9130,
            },
            2: {
                'per_client_target_digit_acc': {0: 0.9107, 1: 0.8945},
                'other_digit_acc': 0.9999,
                'target_subset_acc': 0.0000,
                'other_subset_acc': 0.9985,
                'test_digit_acc': 0.9052,
            },
            3: {
                'per_client_target_digit_acc': {0: 0.9006, 1: 0.8945, 2: 0.9148},
                'other_digit_acc': 0.9996,
                'target_subset_acc': 0.0000,
                'other_subset_acc': 0.9966,
                'test_digit_acc': 0.9025,
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

    print(f"Optimising SSD hyperparameters ({'no-MTL' if is_no_mtl else 'MTL'} case, for client {target_subset_id} ({num_forgotten_clients} forgotten total))")
    current_round = min(max(num_forgotten_clients, 1), 3)
    print_baseline = _make_client_round_baseline(current_round, current_client_id)
    print(f"Target baseline metrics (Round {current_round}, client {current_client_id}): {print_baseline}")

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
        )

        model_state = io.BytesIO()
        torch.save(unlearned_model.state_dict(), model_state)
        trial.set_user_attr("model_state", model_state.getvalue())

        total_delta_score = 0
        
        # --- Stage 1: Evaluate the primary client being forgotten NOW ---
        primary_baseline = _filter_digit_only(_make_client_round_baseline(current_round, current_client_id))
        primary_metrics_tuning = {key: initial_ssd_metrics[key] for key in primary_baseline}
        primary_delta_score = calculate_baseline_delta_score(
            primary_metrics_tuning, primary_baseline, num_forgotten_clients=num_forgotten_clients
        )
        total_delta_score += primary_delta_score

        # --- Stage 2: Evaluate PREVIOUSLY forgotten clients to ensure they REMAIN forgotten ---
        if num_forgotten_clients > 1 and all_forgotten_loaders:
            for client_id, loader in all_forgotten_loaders.items():
                if client_id == target_subset_id:
                    continue 

                # For these clients, compare against the current round's per-client baseline (not accumulating)
                _, subsequent_ssd_metrics = ssd_unlearn_subset(
                    pretrained_model, # The original model is the starting point
                    retain_loader,
                    loader,
                    client_id,
                    device,
                    test_loader=test_loader,
                    calculate_fisher_on=calculate_fisher_on,
                    kill_output_neuron=kill_output_neuron,
                    use_cached_unlearned_model=unlearned_model # Crucially, use the model unlearned in THIS trial
                )
                client_round_baseline = _filter_digit_only(_make_client_round_baseline(current_round, client_id))
                secondary_metrics_tuning = {key: subsequent_ssd_metrics[key] for key in client_round_baseline}
                # Use weights corresponding to a single forgotten client for consistency of scaling
                secondary_delta_score = calculate_baseline_delta_score(
                    secondary_metrics_tuning, client_round_baseline, num_forgotten_clients=1
                )
                total_delta_score += secondary_delta_score

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
    baseline_to_print = _make_client_round_baseline(current_round, current_client_id)
    print("\nBest trial metrics vs baseline for current forgotten client:")
    for key in baseline_to_print:
        current_val = best_trial.user_attrs[key]
        baseline_val = baseline_to_print[key]
        print(f" * {key}: {current_val:.4f} (baseline: {baseline_val:.4f}, diff: {abs(current_val - baseline_val):.4f})")

    output_filename = f"{unlearned_model_name}.h5"
    best_model_state_bytes = best_trial.user_attrs["model_state"]
    with open(output_filename, "wb") as f:
        f.write(best_model_state_bytes)
    print(f"\n✓ Best unlearned model saved to: {output_filename}")

    return study
