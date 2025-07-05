import optuna
from torch.utils.data import DataLoader

from utils import set_global_seed
from ssd import ssd_unlearn_subset
from evaluation import (
    calculate_digit_classification_accuracy,
    calculate_subset_identification_accuracy
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

    The optimisation maximises a scalar score crafted from the requested criteria:

        • minimise train accuracy on forget set (digit & subset)
        • maximise train accuracy on retain set (digit & subset)
        • maximise average test accuracy (digit & subset, all subsets)

    The score is constructed as:

        score =  (
            train_digit_retain  - train_digit_forget
          + train_subset_retain - train_subset_forget
          + test_avg
        )

    Higher score ⇒ better according to the above desiderata.  All terms lie in [0,1].
    """

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

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
        )

        # Compute metrics
        # Train accuracies
        # Forget loader → accuracy on *target* subset (should be data present)
        train_digit_forget, _ = calculate_digit_classification_accuracy(
            unlearned_model, forget_loader, device, target_subset_id
        )
        train_subset_forget, _ = calculate_subset_identification_accuracy(
            unlearned_model, forget_loader, device, target_subset_id
        )

        # Retain loader → accuracy on *other* subsets
        _tmp_tgt, train_digit_retain = calculate_digit_classification_accuracy(
            unlearned_model, retain_loader, device, target_subset_id
        )
        _tmp_tgt2, train_subset_retain = calculate_subset_identification_accuracy(
            unlearned_model, retain_loader, device, target_subset_id
        )

        # Test accuracies (all subsets)
        test_digit_tgt, test_digit_other = calculate_digit_classification_accuracy(
            unlearned_model, test_loader, device, target_subset_id
        )
        test_subset_tgt, test_subset_other = calculate_subset_identification_accuracy(
            unlearned_model, test_loader, device, target_subset_id
        )
        test_avg = (
            test_digit_tgt + test_digit_other
        ) / 4.0

        # Scalar objective
        score = (
            (train_digit_retain - train_digit_forget)
            + (train_subset_retain - train_subset_forget)
            + test_avg
        )

        # Verbose output so the user can monitor per-trial metrics (including TEST accuracies)
        print(
            f"[Trial {trial.number:03d}] α={alpha:.4f}, λ={lam:.4f} | "
            f"Train-Fgt Dig={train_digit_forget:.4f}, Train-Ret Dig={train_digit_retain:.4f} | "
            f"Train-Fgt Sub={train_subset_forget:.4f}, Train-Ret Sub={train_subset_retain:.4f} | "
            f"[TEST] Dig-Tgt={test_digit_tgt:.4f}, Dig-Oth={test_digit_other:.4f}, "
            f"Sub-Tgt={test_subset_tgt:.4f}, Sub-Oth={test_subset_other:.4f} | "
            f"Score={score:.4f}"
        )

        # Keep track for analysis
        trial.set_user_attr("train_digit_forget", train_digit_forget)
        trial.set_user_attr("train_digit_retain", train_digit_retain)
        trial.set_user_attr("train_subset_forget", train_subset_forget)
        trial.set_user_attr("train_subset_retain", train_subset_retain)
        trial.set_user_attr("test_avg", test_avg)

        return score

    study.optimize(_objective, n_trials=n_trials, show_progress_bar=True)

    print("\nOptuna optimisation completed.")
    print("Best score: {:.4f}".format(study.best_value))
    print("Best hyper-parameters (α, λ):", study.best_params)

    return study 