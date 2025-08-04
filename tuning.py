import optuna
import torch
from torch.utils.data import DataLoader
from typing import Optional
import io

from utils import set_global_seed, DEFAULT_BASELINE_METRICS, calculate_baseline_delta_score
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
):
    """Run TPE search to tune SSD hyper-parameters α (exponent) and λ (dampening_constant).

    For MTL case (target_subset_id is not None):
        The optimisation minimises the distance to baseline metrics:
        • target_digit_acc: 0.9056
        • other_digit_acc: 0.9998
        • target_subset_acc: 0.0000
        • test_digit_acc: 0.9130

    For no-MTL case (target_subset_id is None):
        The optimisation minimises the distance to baseline metrics:
        • target_digit_acc: 0.9000 (Train accuracy on target subset)
        • other_digit_acc: 0.9900 (Train accuracy on other subsets)  
        • test_digit_acc: 0.9100 (Test Accuracy)

    The objective minimises the weighted absolute difference between current metrics
    and baseline metrics. Lower score ⇒ closer to baseline performance.
    """

    # Check if this is the no-MTL case
    is_no_mtl = (target_subset_id is None)
    
    if is_no_mtl:
        # No-MTL baseline metrics (from train_baseline_all_no_mtl.py results)
        BASELINE_METRICS_TUNING = {
            'target_digit_acc': 0.8902,  # Accuracy on target subset 
            'other_digit_acc': 0.9999,   # Accuracy on other subsets (training data)
            'test_digit_acc': 0.9061,    # Test accuracy
        }
    else:
        # MTL baseline metrics (from baseline.py results)  
        BASELINE_METRICS_TUNING = {
            'target_digit_acc': 0.8956,  # Accuracy on target subset
            'other_digit_acc': 0.9998,   # Accuracy on other subsets
            #'target_subset_acc': 0.0000, # Subset ID accuracy on target (should be 0 after unlearning)
            'other_subset_acc': 0.9974,  # Subset ID accuracy on other subsets
            'test_digit_acc': 0.9130,    # Test accuracy
        }

    print(f"Optimising SSD hyperparameters ({'no-MTL' if is_no_mtl else 'MTL'} case)")
    print(f"Target baseline metrics: {BASELINE_METRICS_TUNING}")

    # Set up Optuna sampler with fixed seed for reproducibility
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=10, n_ei_candidates=24)
    
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def _objective(trial):
        alpha = trial.suggest_float("alpha", 0.1, 100.0, log=True)
        lam = trial.suggest_float("lambda", 0.01, 10.0, log=True)

        # Apply SSD unlearning with the proposed hyper-parameters
        unlearned_model, ssd_metrics = ssd_unlearn_subset(
            pretrained_model,
            retain_loader,
            forget_loader,
            target_subset_id,
            device,
            lower_bound=1.0,              # keep default
            exponent=alpha,
            dampening_constant=lam,
            selection_weighting=1.0,
            test_loader=test_loader,      # pass test_loader to get test accuracy
            calculate_fisher_on=calculate_fisher_on,
        )

        # Store the model state in memory to avoid saving to disk every trial
        model_state = io.BytesIO()
        torch.save(unlearned_model.state_dict(), model_state)
        trial.set_user_attr("model_state", model_state.getvalue())


        # Calculate ALL metrics from SSD results
        current_metrics_all = {
            'target_digit_acc': ssd_metrics['target_digit_acc'],
            'other_digit_acc': ssd_metrics['other_digit_acc'],
            'target_subset_acc': ssd_metrics['target_subset_acc'],
            'other_subset_acc': ssd_metrics['other_subset_acc'],
            'test_digit_acc': ssd_metrics['test_digit_acc'],
            'mia_score': ssd_metrics['mia_score']
        }

        # Use only the relevant metrics for delta score calculation
        current_metrics_tuning = {}
        for key in BASELINE_METRICS_TUNING:
            current_metrics_tuning[key] = ssd_metrics[key]

        # Calculate distance to baseline metrics (lower is better)
        delta_score = calculate_baseline_delta_score(current_metrics_tuning, BASELINE_METRICS_TUNING)

        # Log trial results for debugging
        trial.set_user_attr("target_digit_acc", current_metrics_all['target_digit_acc'])
        trial.set_user_attr("other_digit_acc", current_metrics_all['other_digit_acc'])
        trial.set_user_attr("test_digit_acc", current_metrics_all['test_digit_acc'])
        trial.set_user_attr("mia_score", current_metrics_all['mia_score'])
        if not is_no_mtl:
            trial.set_user_attr("target_subset_acc", current_metrics_all['target_subset_acc'])
            trial.set_user_attr("other_subset_acc", current_metrics_all['other_subset_acc'])

        return delta_score

    study.optimize(_objective, n_trials=n_trials)

    # Print detailed results
    print(f"\n--- SSD Hyperparameter Optimisation Finished ({'no-MTL' if is_no_mtl else 'MTL'} case) ---")
    print(f"Best α (exponent): {study.best_params['alpha']:.6f}")
    print(f"Best λ (dampening): {study.best_params['lambda']:.6f}")
    print(f"Best objective value (delta score): {study.best_value:.6f}")
    
    print("\nBest trial metrics vs baseline:")
    print("(* indicates metrics used for optimization)")
    best_trial = study.best_trial
    for key in BASELINE_METRICS_TUNING:
        current_val = best_trial.user_attrs[key]
        baseline_val = BASELINE_METRICS_TUNING[key]
        diff = abs(current_val - baseline_val)
        print(f" * {key}: {current_val:.4f} (baseline: {baseline_val:.4f}, diff: {diff:.4f})")
    
    # Print non-optimized metrics for reference
    if not is_no_mtl:
        mia_score = best_trial.user_attrs['mia_score']
        print(f"   mia_score: {mia_score:.2f}%")
    else:
        target_subset_acc = best_trial.user_attrs.get('target_subset_acc', 0.0)
        other_subset_acc = best_trial.user_attrs.get('other_subset_acc', 0.0)
        mia_score = best_trial.user_attrs['mia_score']
        print(f"   target_subset_acc: {target_subset_acc:.4f}")
        print(f"   other_subset_acc: {other_subset_acc:.4f}")  
        print(f"   mia_score: {mia_score:.2f}%")

    # Save the best model
    best_model_state_bytes = best_trial.user_attrs["model_state"]
    best_model_state = io.BytesIO(best_model_state_bytes)
    best_model_state.seek(0)
    
    # We need to load it into the original model structure
    # The `pretrained_model` is a good template for this.
    final_model = pretrained_model
    final_model.load_state_dict(torch.load(best_model_state, map_location=device))
    
    output_filename = "unlearned_model_mtl.h5" if not is_no_mtl else "unlearned_model_no_mtl.h5"
    torch.save(final_model.state_dict(), output_filename)
    print(f"\n✓ Best unlearned model saved to: {output_filename}")

    return study 