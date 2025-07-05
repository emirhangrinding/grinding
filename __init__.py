"""Multi-Task Learning with DISSOLVE and SSD Unlearning Package

This package provides:
- Multi-task learning with two heads (digit classification and subset identification)
- DISSOLVE unlearning method
- Selective Synaptic Dampening (SSD) unlearning
- Baseline training excluding one client
- Hyperparameter optimization for SSD
- Data preparation and visualization utilities
"""

__version__ = "1.0.0"
__author__ = "Multi-Task Learning Team"

# Import main functions for easy access
from .training import learn, train_mtl_two_heads
from .dissolve import dissolve_unlearn_subset
from .ssd import ssd_unlearn_subset
from .baseline import learn_baseline_excluding_client
from .evaluation import (
    calculate_digit_classification_accuracy,
    calculate_subset_identification_accuracy,
    get_membership_attack_prob_train_only
)
from .models import MTL_Two_Heads_ResNet
from .utils import set_global_seed
from .visualization import visualize_mtl_two_heads_results

__all__ = [
    "learn",
    "train_mtl_two_heads", 
    "dissolve_unlearn_subset",
    "ssd_unlearn_subset",
    "learn_baseline_excluding_client",
    "calculate_digit_classification_accuracy",
    "calculate_subset_identification_accuracy", 
    "get_membership_attack_prob_train_only",
    "MTL_Two_Heads_ResNet",
    "set_global_seed",
    "visualize_mtl_two_heads_results"
] 