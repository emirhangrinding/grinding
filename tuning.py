import optuna
from torch.utils.data import DataLoader

from utils import set_global_seed, DEFAULT_BASELINE_METRICS, calculate_baseline_delta_score
from ssd import ssd_unlearn_subset
from evaluation import (
    calculate_digit_classification_accuracy,
    calculate_subset_identification_accuracy,
    get_membership_attack_prob_train_only
)

def optimise_ssd_hyperparams(
    pretrained_model,
    retain_loader,
    forget_loader,
    test_loader,
    device,
    target_subset_id: int,
    n_trials: int = 25,
    seed: int = 42,
):
    """Run TPE search to tune SSD hyper-parameters α (exponent) and λ (dampening_constant).

    The optimisation minimises the distance to baseline metrics:
        • target_digit_acc: 0.9056
        • other_digit_acc: 0.9998
        • target_subset_acc: 0.0000
        • other_subset_acc: 0.9974
        • test_digit_acc: 0.9130
        • mia_score: 75.44

    The objective minimises the weighted absolute difference between current metrics
    and baseline metrics. Lower score ⇒ closer to baseline performance.
    """

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def _objective(trial):
        alpha = trial.suggest_float("alpha", 0.1, 100.0, log=True)
        lam = trial.suggest_float("lambda", 0.1, 5.0, log=True)

        # Apply SSD unlearning with the proposed hyper-parameters
        unlearned_model = ssd_unlearn_subset(
            pretrained_model,
            retain_loader,
            forget_loader,
            target_subset_id,
            device,
            lower_bound=1.0,              # keep default
            exponent=alpha,
            dampening_constant=lam,
            selection_weighting=1.0,
            test_loader=None,             # we compute custom metrics below to avoid duplicate work
            calculate_fisher_on="subset",  # default to subset task for tuning
        )

        # Compute test accuracies to match baseline metric format
        test_digit_tgt, test_digit_other = calculate_digit_classification_accuracy(
            unlearned_model, test_loader, device, target_subset_id
        )
        test_subset_tgt, test_subset_other = calculate_subset_identification_accuracy(
            unlearned_model, test_loader, device, target_subset_id
        )
        
        # Calculate overall test digit accuracy (average of target and other)
        test_digit_overall = (test_digit_tgt + test_digit_other) / 2.0

        # Calculate MIA score
        mia_score = get_membership_attack_prob_train_only(retain_loader, forget_loader, unlearned_model)

        # Format current metrics to match baseline structure
        current_metrics = {
            'target_digit_acc': test_digit_tgt,
            'other_digit_acc': test_digit_other,
            'target_subset_acc': test_subset_tgt,
            'other_subset_acc': test_subset_other,
            'test_digit_acc': test_digit_overall,
            'mia_score': mia_score
        }

        # Calculate distance to baseline metrics (lower is better)
        delta_score = calculate_baseline_delta_score(current_metrics, DEFAULT_BASELINE_METRICS)

        # Verbose output so the user can monitor per-trial metrics
        print(
            f"[Trial {trial.number:03d}] α={alpha:.4f}, λ={lam:.4f} | "
            f"Target Digit={test_digit_tgt:.4f} (baseline: {DEFAULT_BASELINE_METRICS['target_digit_acc']:.4f}), "
            f"Other Digit={test_digit_other:.4f} (baseline: {DEFAULT_BASELINE_METRICS['other_digit_acc']:.4f}), "
            f"Target Subset={test_subset_tgt:.4f} (baseline: {DEFAULT_BASELINE_METRICS['target_subset_acc']:.4f}), "
            f"Other Subset={test_subset_other:.4f} (baseline: {DEFAULT_BASELINE_METRICS['other_subset_acc']:.4f}), "
            f"Test Digit Overall={test_digit_overall:.4f} (baseline: {DEFAULT_BASELINE_METRICS['test_digit_acc']:.4f}), "
            f"MIA={mia_score:.2f}% (baseline: {DEFAULT_BASELINE_METRICS['mia_score']:.2f}%) | "
            f"Delta Score={delta_score:.4f}"
        )

        # Keep track for analysis
        trial.set_user_attr("target_digit_acc", test_digit_tgt)
        trial.set_user_attr("other_digit_acc", test_digit_other)
        trial.set_user_attr("target_subset_acc", test_subset_tgt)
        trial.set_user_attr("other_subset_acc", test_subset_other)
        trial.set_user_attr("test_digit_acc", test_digit_overall)
        trial.set_user_attr("mia_score", mia_score)
        trial.set_user_attr("delta_score", delta_score)

        return delta_score

    study.optimize(_objective, n_trials=n_trials, show_progress_bar=True)

    print("\nOptuna optimisation completed.")
    print("Best delta score (closest to baseline): {:.4f}".format(study.best_value))
    print("Best hyper-parameters (α, λ):", study.best_params)
    
    # Print best metrics vs baseline
    best_trial = study.best_trial
    print("\nBest trial metrics vs baseline:")
    metrics_to_show = ['target_digit_acc', 'other_digit_acc', 'target_subset_acc', 'other_subset_acc', 'test_digit_acc', 'mia_score']
    for metric in metrics_to_show:
        current_val = best_trial.user_attrs[metric]
        baseline_val = DEFAULT_BASELINE_METRICS[metric]
        if metric == 'mia_score':
            print(f"  {metric}: {current_val:.2f}% (baseline: {baseline_val:.2f}%, diff: {abs(current_val - baseline_val):.2f}%)")
        else:
            print(f"  {metric}: {current_val:.4f} (baseline: {baseline_val:.4f}, diff: {abs(current_val - baseline_val):.4f})")

    return study 