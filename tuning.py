import optuna
from torch.utils.data import DataLoader

from utils import set_global_seed, DEFAULT_BASELINE_METRICS, calculate_baseline_delta_score
from ssd import ssd_unlearn_subset

def optimise_ssd_hyperparams(
    pretrained_model,
    retain_loader,
    forget_loader,
    test_loader,
    device,
    target_subset_id: int,
    n_trials: int = 25,
    seed: int = 42,
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
        • target_digit_acc: 0.9000 (Train accuracy on target)
        • other_digit_acc: 0.9900 (Train accuracy on digit)
        • test_digit_acc: 0.9100 (Test Accuracy)

    The objective minimises the weighted absolute difference between current metrics
    and baseline metrics. Lower score ⇒ closer to baseline performance.
    """

    # Check if this is the no-MTL case
    is_no_mtl = target_subset_id is None

    if is_no_mtl:
        # No-MTL baselines as specified by user
        BASELINE_METRICS_WITH_MIA = {
            'target_digit_acc': 0.9000,  # Train accuracy on target: 90.00%
            'other_digit_acc': 0.9900,   # Train accuracy on digit: 99.00%
            'target_subset_acc': 0.0000,  # Not meaningful in no-MTL, but kept for compatibility
            'other_subset_acc': 0.0000,   # Not meaningful in no-MTL, but kept for compatibility
            'test_digit_acc': 0.9100,     # Test Accuracy: 91.00%
            'mia_score': 75.44             # Keep same MIA baseline for now
        }

        # Only the relevant metrics for optimization in no-MTL case
        BASELINE_METRICS_TUNING = {
            'target_digit_acc': 0.9000,  # Train accuracy on target: 90.00%
            'other_digit_acc': 0.9900,   # Train accuracy on digit: 99.00%
            'test_digit_acc': 0.9100     # Test Accuracy: 91.00%
        }
        print("Using no-MTL baselines:")
        print("  - Train accuracy on target: 90.00%")
        print("  - Train accuracy on digit: 99.00%")
        print("  - Test Accuracy: 91.00%")
    else:
        # MTL baselines (original)
        BASELINE_METRICS_WITH_MIA = DEFAULT_BASELINE_METRICS.copy()
        BASELINE_METRICS_WITH_MIA['mia_score'] = 75.44

        # Only the specified metrics for optimization in MTL case
        BASELINE_METRICS_TUNING = {
            'target_digit_acc': 0.9056,
            'other_digit_acc': 0.9998,
            'target_subset_acc': 0.0000,
            'test_digit_acc': 0.9130
        }
        print("Using MTL baselines:")
        print("  - Target digit accuracy: 90.56%")
        print("  - Other digit accuracy: 99.98%")
        print("  - Target subset accuracy: 0.00%")
        print("  - Test digit accuracy: 91.30%")

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def _objective(trial):
        alpha = trial.suggest_float("alpha", 0.1, 100.0, log=True)
        lam = trial.suggest_float("lambda", 0.1, 5.0, log=True)

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

        # Verbose output showing ALL metrics for monitoring
        if is_no_mtl:
            print(
                f"[Trial {trial.number:03d}] α={alpha:.4f}, λ={lam:.4f} | "
                f"Target Digit={current_metrics_all['target_digit_acc']:.4f} (baseline: {BASELINE_METRICS_WITH_MIA['target_digit_acc']:.4f}), "
                f"Other Digit={current_metrics_all['other_digit_acc']:.4f} (baseline: {BASELINE_METRICS_WITH_MIA['other_digit_acc']:.4f}), "
                f"Test Digit Overall={current_metrics_all['test_digit_acc']:.4f} (baseline: {BASELINE_METRICS_WITH_MIA['test_digit_acc']:.4f}), "
                f"MIA={current_metrics_all['mia_score']:.2f}% (baseline: {BASELINE_METRICS_WITH_MIA['mia_score']:.2f}%) | "
                f"Delta Score={delta_score:.4f} (optimizing on 3 metrics: target_digit_acc, other_digit_acc, test_digit_acc)"
            )
        else:
            print(
                f"[Trial {trial.number:03d}] α={alpha:.4f}, λ={lam:.4f} | "
                f"Target Digit={current_metrics_all['target_digit_acc']:.4f} (baseline: {BASELINE_METRICS_WITH_MIA['target_digit_acc']:.4f}), "
                f"Other Digit={current_metrics_all['other_digit_acc']:.4f} (baseline: {BASELINE_METRICS_WITH_MIA['other_digit_acc']:.4f}), "
                f"Target Subset={current_metrics_all['target_subset_acc']:.4f} (baseline: {BASELINE_METRICS_WITH_MIA['target_subset_acc']:.4f}), "
                f"Other Subset={current_metrics_all['other_subset_acc']:.4f} (baseline: {BASELINE_METRICS_WITH_MIA['other_subset_acc']:.4f}), "
                f"Test Digit Overall={current_metrics_all['test_digit_acc']:.4f} (baseline: {BASELINE_METRICS_WITH_MIA['test_digit_acc']:.4f}), "
                f"MIA={current_metrics_all['mia_score']:.2f}% (baseline: {BASELINE_METRICS_WITH_MIA['mia_score']:.2f}%) | "
                f"Delta Score={delta_score:.4f} (optimizing on 4 metrics: target_digit_acc, other_digit_acc, target_subset_acc, test_digit_acc)"
            )

        # Keep track of ALL metrics for analysis
        trial.set_user_attr("target_digit_acc", current_metrics_all['target_digit_acc'])
        trial.set_user_attr("other_digit_acc", current_metrics_all['other_digit_acc'])
        trial.set_user_attr("target_subset_acc", current_metrics_all['target_subset_acc'])
        trial.set_user_attr("other_subset_acc", current_metrics_all['other_subset_acc'])
        trial.set_user_attr("test_digit_acc", current_metrics_all['test_digit_acc'])
        trial.set_user_attr("mia_score", current_metrics_all['mia_score'])
        trial.set_user_attr("delta_score", delta_score)

        return delta_score

    study.optimize(_objective, n_trials=n_trials, show_progress_bar=True)

    print("\nOptuna optimisation completed.")
    num_opt_metrics = len(BASELINE_METRICS_TUNING)
    print(f"Best delta score (closest to baseline on {num_opt_metrics} metrics): {study.best_value:.4f}")
    print("Best hyper-parameters (α, λ):", study.best_params)
    
    # Print ALL metrics vs baseline for the best trial
    best_trial = study.best_trial
    print("\nBest trial metrics vs baseline:")
    print("(* indicates metrics used for optimization)")
    
    if is_no_mtl:
        metrics_to_show = ['target_digit_acc', 'other_digit_acc', 'test_digit_acc', 'mia_score']
        optimization_metrics = set(BASELINE_METRICS_TUNING.keys())
    else:
        metrics_to_show = ['target_digit_acc', 'other_digit_acc', 'target_subset_acc', 'other_subset_acc', 'test_digit_acc', 'mia_score']
        optimization_metrics = set(BASELINE_METRICS_TUNING.keys())
    
    for metric in metrics_to_show:
        current_val = best_trial.user_attrs[metric]
        baseline_val = BASELINE_METRICS_WITH_MIA[metric]
        star = "*" if metric in optimization_metrics else " "
        if metric == 'mia_score':
            print(f" {star} {metric}: {current_val:.2f}% (baseline: {baseline_val:.2f}%, diff: {abs(current_val - baseline_val):.2f}%)")
        else:
            print(f" {star} {metric}: {current_val:.4f} (baseline: {baseline_val:.4f}, diff: {abs(current_val - baseline_val):.4f})")

    return study 