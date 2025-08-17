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

    # Define baseline metrics for different stages (Rounds 1-3)
    if baseline_variant == "no_mtl":
        # No-MTL baselines (baseline_all_clients_model.h5)
        BASELINE_METRICS_ROUND_1 = {
            'target_digit_acc': 0.8983, 'other_digit_acc': 0.9999, 'test_digit_acc': 0.9061
        }
        BASELINE_METRICS_ROUND_2 = {
            'target_digit_acc': 0.8784, 'other_digit_acc': 1.0000, 'test_digit_acc': 0.8931
        }
        BASELINE_METRICS_ROUND_3 = {
            'target_digit_acc': 0.8912, 'other_digit_acc': 0.9999, 'test_digit_acc': 0.8919
        }
    elif baseline_variant == "mtl_ce":
        # MTL (CE-only) baselines (baseline_mtl_all_clients_no_dis.h5)
        BASELINE_METRICS_ROUND_1 = {
            'target_digit_acc': 0.8975, 'other_digit_acc': 0.9995,
            'target_subset_acc': 0.0000, 'other_subset_acc': 0.9909, 'test_digit_acc': 0.9042
        }
        BASELINE_METRICS_ROUND_2 = {
            'target_digit_acc': 0.8843, 'other_digit_acc': 0.9995,
            'target_subset_acc': 0.0000, 'other_subset_acc': 0.9940, 'test_digit_acc': 0.8986
        }
        BASELINE_METRICS_ROUND_3 = {
            'target_digit_acc': 0.8983, 'other_digit_acc': 0.9981,
            'target_subset_acc': 0.0000, 'other_subset_acc': 0.9735, 'test_digit_acc': 0.8869
        }
    else:
        # Standard MTL baselines (baseline_mtl_all_clients.h5)
        BASELINE_METRICS_ROUND_1 = {
            'target_digit_acc': 0.9037, 'other_digit_acc': 0.9998,
            'target_subset_acc': 0.0000, 'other_subset_acc': 0.9974, 'test_digit_acc': 0.9130
        }
        BASELINE_METRICS_ROUND_2 = {
            'target_digit_acc': 0.8945, 'other_digit_acc': 0.9999,
            'target_subset_acc': 0.0000, 'other_subset_acc': 0.9985, 'test_digit_acc': 0.9052
        }
        BASELINE_METRICS_ROUND_3 = {
            'target_digit_acc': 0.8993, 'other_digit_acc': 0.9981,
            'target_subset_acc': 0.0000, 'other_subset_acc': 0.9735, 'test_digit_acc': 0.8969
        }

    # Optionally ignore subset-ID metrics entirely for tuning objective
    if digit_metrics_only:
        def _filter_digit_only(metrics_dict):
            allowed = {'target_digit_acc', 'other_digit_acc', 'test_digit_acc'}
            return {k: v for k, v in metrics_dict.items() if k in allowed}
        BASELINE_METRICS_ROUND_1 = _filter_digit_only(BASELINE_METRICS_ROUND_1)
        BASELINE_METRICS_ROUND_2 = _filter_digit_only(BASELINE_METRICS_ROUND_2)
        # Round 3 may not exist in some variants; guard accordingly
        try:
            BASELINE_METRICS_ROUND_3 = _filter_digit_only(BASELINE_METRICS_ROUND_3)
        except NameError:
            pass

    print(f"Optimising SSD hyperparameters ({'no-MTL' if is_no_mtl else 'MTL'} case, for client {target_subset_id} ({num_forgotten_clients} forgotten total))")
    if num_forgotten_clients == 1:
        print(f"Target baseline metrics (Round 1): {BASELINE_METRICS_ROUND_1}")
    elif num_forgotten_clients == 2:
        print(f"Target baseline metrics (Round 2): {BASELINE_METRICS_ROUND_2}")
        print(f"Also ensuring Round 1 metrics are maintained for previously forgotten clients.")
    else:
        print(f"Target baseline metrics (Round 3): {BASELINE_METRICS_ROUND_3}")
        print(f"Also ensuring Round 1 metrics are maintained for previously forgotten clients.")

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
        if num_forgotten_clients == 1:
            primary_baseline = BASELINE_METRICS_ROUND_1
        elif num_forgotten_clients == 2:
            primary_baseline = BASELINE_METRICS_ROUND_2
        else:
            primary_baseline = BASELINE_METRICS_ROUND_3
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

                # For these clients, we check against Round 1 baseline
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
                
                secondary_metrics_tuning = {key: subsequent_ssd_metrics[key] for key in BASELINE_METRICS_ROUND_1}
                # Weighting for these is for 1 forgotten client
                secondary_delta_score = calculate_baseline_delta_score(
                    secondary_metrics_tuning, BASELINE_METRICS_ROUND_1, num_forgotten_clients=1
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
    if num_forgotten_clients == 1:
        baseline_to_print = BASELINE_METRICS_ROUND_1
    elif num_forgotten_clients == 2:
        baseline_to_print = BASELINE_METRICS_ROUND_2
    else:
        baseline_to_print = BASELINE_METRICS_ROUND_3
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
