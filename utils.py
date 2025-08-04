import random
import numpy as np
import torch
import torch.nn as nn

# Global seed utility for reproducibility
SEED_DEFAULT = 42

# Default baseline metrics for unlearning experiments
DEFAULT_BASELINE_METRICS = {
    'target_digit_acc': 0.9056,
    'other_digit_acc': 0.9998, 
    'target_subset_acc': 0.0000,
    'other_subset_acc': 0.9974,
    'test_digit_acc': 0.9130
}

def set_global_seed(seed: int = SEED_DEFAULT):
    """Set random seeds for Python, NumPy and PyTorch to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensuring determinism for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def pull_to_means(z, y):
    """
    z: [N, D] tensor of embeddings
    y: [N] tensor of class labels (e.g., y2)
    
    Returns average distance of points to their class mean.
    """
    unique_classes = y.unique()
    loss = 0.0
    for cls in unique_classes:
        idx = (y == cls)
        if idx.sum() < 2:
            continue
        class_embeddings = z[idx]                      # [n_c, D]
        class_mean = class_embeddings.mean(dim=0)      # [D]
        distances = (class_embeddings - class_mean).pow(2).sum(dim=1)  # [n_c]
        loss += distances.mean()
    return loss / len(unique_classes)

def push_from_means(z, y, margin=1.0):
    """
    z: [N, D] embeddings
    y: [N] class labels (e.g., y2)
    Returns a loss that encourages class means to be far apart.
    """
    unique_classes = y.unique()
    class_means = []
    
    for cls in unique_classes:
        idx = (y == cls)
        if idx.sum() < 2:
            continue
        class_embeddings = z[idx]
        class_mean = class_embeddings.mean(dim=0)
        class_means.append(class_mean)
    
    if len(class_means) < 2:
        return torch.tensor(0.0, device=z.device)  # No repulsion possible

    # Stack all means: [K, D]
    means = torch.stack(class_means, dim=0)
    # Compute pairwise distances
    dists = torch.cdist(means, means, p=2)  # [K, K]
    
    # We only care about unique pairs (i < j)
    num_classes = means.size(0)
    push_loss = 0.0
    count = 0

    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            dist = dists[i, j]
            if dist < margin:
                push_loss += (margin - dist).pow(2)
                count += 1

    if count == 0:
        return torch.tensor(0.0, device=z.device)
    
    return push_loss / count

def distance_to_mean_loss(z, y):
    """
    z: [N, D] tensor of embeddings
    y: [N] tensor of class labels (e.g., y2)

    Returns average distance of points to their class mean.
    """
    return pull_to_means(z, y)

def intra_y1_y2_disentanglement_loss(z, y1, y2, lambda_pull=1.0, lambda_push=1.0):
    """
    For each y1 class, compute pull and push losses using y2 as the grouping label.
    This encourages within-class subset separation while maintaining cross-class structure.
    """
    loss = 0.0
    margin = 1.0  # Hinge loss margin
    unique_y1 = y1.unique()
    for y1_class in unique_y1:
        idx = (y1 == y1_class)
        if idx.sum() < 2:
            continue
        z_group = z[idx]
        y2_group = y2[idx]
        pull_loss = pull_to_means(z_group, y2_group)
        push_loss = push_from_means(z_group, y2_group, margin=margin)
        loss += lambda_pull * pull_loss + lambda_push * push_loss

    return loss / len(unique_y1)

def dirichlet_partition(dataset, num_clients=10, alpha=0.6, num_classes=10, seed=42):
    """Partition dataset indices into non-IID client splits via Dirichlet distribution."""
    if seed is not None:
        np.random.seed(seed)
    # Get labels for all samples
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    # For each class, partition its indices
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    client_indices = [[] for _ in range(num_clients)]
    for c, idxs in enumerate(class_indices):
        # Dirichlet draw
        proportions = np.random.dirichlet([alpha]*num_clients)
        # Compute how many samples each client gets from class c
        splits = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
        split_idxs = np.split(idxs, splits)
        for i, client_idx in enumerate(client_indices):
            client_idx.extend(split_idxs[i].tolist())
    return client_indices

def calculate_baseline_delta_score(current_metrics, baseline_metrics, weights=None):
    """
    Calculate how close current metrics are to baseline metrics using weighted absolute differences.
    
    Args:
        current_metrics: dict with keys: 'target_digit_acc', 'other_digit_acc', 
                        'target_subset_acc', 'other_subset_acc', 'test_digit_acc'
        baseline_metrics: dict with same keys as current_metrics
        weights: dict with same keys for weighting different metrics (optional)
    
    Returns:
        float: delta score (lower is closer to baseline)
    """
    if weights is None:
        # Default equal weights for all metrics
        weights = {
            'target_digit_acc': 3.0,
            'other_digit_acc': 1.0, 
            'target_subset_acc': 1.0,
            'other_subset_acc': 1.0,
            'test_digit_acc': 1.0
        }
    
    delta = 0.0
    total_weight = 0.0
    
    for key in current_metrics:
        if key in baseline_metrics and key in weights:
            delta += weights[key] * abs(current_metrics[key] - baseline_metrics[key])
            total_weight += weights[key]
    
    # Normalize by total weight to get average weighted delta
    return delta / total_weight if total_weight > 0 else float('inf')

def track_best_epoch_vs_baseline(epoch, current_metrics, baseline_metrics, best_epoch_info, weights=None):
    """
    Track which epoch produces results closest to baseline metrics.
    
    Args:
        epoch: current epoch number (1-indexed)
        current_metrics: dict of current epoch metrics
        baseline_metrics: dict of baseline metrics to compare against
        best_epoch_info: dict to store best epoch info (modified in-place)
        weights: optional dict of metric weights
    
    Returns:
        tuple: (current_delta_score, is_best_so_far)
    """
    current_delta = calculate_baseline_delta_score(current_metrics, baseline_metrics, weights)
    
    is_best = False
    if 'best_delta' not in best_epoch_info or current_delta < best_epoch_info['best_delta']:
        best_epoch_info['best_delta'] = current_delta
        best_epoch_info['best_epoch'] = epoch
        best_epoch_info['best_metrics'] = current_metrics.copy()
        is_best = True
    
    return current_delta, is_best 