import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
import matplotlib


# Use non-interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


try:
    import open_clip  # type: ignore

    _HAS_OPEN_CLIP = True
except Exception:  # pragma: no cover - optional dep
    _HAS_OPEN_CLIP = False

try:
    import clip as openai_clip  # type: ignore

    _HAS_OPENAI_CLIP = True
except Exception:  # pragma: no cover - optional dep
    _HAS_OPENAI_CLIP = False

from torchvision import datasets, transforms


RNG = np.random.RandomState


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_results_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def human_size(num: int) -> str:
    for unit in ["", "K", "M", "G"]:
        if abs(num) < 1000:
            return f"{num:.1f}{unit}"
        num /= 1000.0
    return f"{num:.1f}T"


def create_dirichlet_partitions(
    y: Sequence[int],
    num_clients: int,
    alpha: float,
    num_classes: int,
    min_size_per_client: int,
    seed: int,
) -> List[List[int]]:
    """
    Create non-IID client partitions using Dirichlet distribution across classes.

    Ensures each client gets at least `min_size_per_client` samples.
    """
    rng = RNG(seed)
    indices_per_class = [np.where(np.array(y) == k)[0] for k in range(num_classes)]

    attempt = 0
    while True:
        attempt += 1
        client_indices: List[List[int]] = [[] for _ in range(num_clients)]
        for k in range(num_classes):
            idx_k = indices_per_class[k].copy()
            rng.shuffle(idx_k)
            proportions = rng.dirichlet([alpha] * num_clients)

            # Optional balancing to reduce extreme skews
            # This discourages assigning to already-large clients
            sizes = np.array([len(ci) for ci in client_indices], dtype=np.float64) + 1e-6
            inv_sizes = 1.0 / sizes
            proportions = proportions * inv_sizes
            proportions = proportions / proportions.sum()

            splits = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            for j, part in enumerate(np.split(idx_k, splits)):
                client_indices[j].extend(part.tolist())

        sizes_now = [len(ci) for ci in client_indices]
        if min(sizes_now) >= min_size_per_client:
            break
        if attempt >= 20:
            # Relax requirement to avoid infinite loop in extreme cases
            break

    return client_indices


@dataclass
class IndexedSample:
    index: int
    class_label: int
    client_id: int


class IndexedSubset(data.Dataset):
    def __init__(
        self,
        base_dataset: data.Dataset,
        indexed_samples: List[IndexedSample],
        transform: transforms.Compose,
    ) -> None:
        self.base_dataset = base_dataset
        self.indexed_samples = indexed_samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indexed_samples)

    def __getitem__(self, i: int):
        s = self.indexed_samples[i]
        img, target = self.base_dataset[s.index]
        if self.transform is not None:
            img = self.transform(img)
        return img, target, s.client_id, s.index


def build_indexed_samples(
    targets: Sequence[int],
    client_indices: List[List[int]],
    max_samples_total: int | None,
    seed: int,
) -> List[IndexedSample]:
    rng = RNG(seed)
    per_client_lists: List[List[int]] = []
    for cid, idxs in enumerate(client_indices):
        rng.shuffle(idxs)
        per_client_lists.append(idxs)

    all_indices: List[Tuple[int, int]] = []  # (idx, client_id)
    if max_samples_total is None or max_samples_total <= 0:
        for cid, idxs in enumerate(per_client_lists):
            all_indices.extend([(idx, cid) for idx in idxs])
    else:
        # Sample proportionally to client sizes
        sizes = np.array([len(idxs) for idxs in per_client_lists], dtype=np.float64)
        probs = sizes / sizes.sum()
        samples_per_client = np.random.multinomial(max_samples_total, probs)
        for cid, (idxs, k) in enumerate(zip(per_client_lists, samples_per_client)):
            chosen = idxs[: int(k)] if k <= len(idxs) else idxs
            all_indices.extend([(idx, cid) for idx in chosen])

    rng.shuffle(all_indices)
    indexed = [IndexedSample(index=idx, class_label=int(targets[idx]), client_id=cid) for idx, cid in all_indices]
    return indexed


def load_clip_model(
    model_name: str,
    device: torch.device,
) -> Tuple[nn.Module, transforms.Compose, Dict[str, str]]:
    meta: Dict[str, str] = {}
    if _HAS_OPEN_CLIP:
        # Prefer OpenCLIP for wider pretrained coverage
        pretrained = "laion2b_s34b_b79k"
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
        model.eval()
        meta["provider"] = "open_clip"
        meta["pretrained"] = pretrained
        return model, preprocess, meta
    if _HAS_OPENAI_CLIP:
        model, preprocess = openai_clip.load(model_name, device=device)
        model.eval()
        meta["provider"] = "openai_clip"
        meta["pretrained"] = "openai"
        return model, preprocess, meta
    raise ImportError(
        "Neither 'open_clip_torch' nor 'clip' is installed. Please install one of them: \n"
        "  pip install open_clip_torch  OR  pip install git+https://github.com/openai/CLIP.git"
    )


@torch.no_grad()
def encode_features(
    model: nn.Module,
    loader: data.DataLoader,
    device: torch.device,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feats: List[np.ndarray] = []
    class_labels: List[np.ndarray] = []
    client_ids: List[np.ndarray] = []
    indices: List[np.ndarray] = []

    for images, y, cids, idxs in loader:
        images = images.to(device, non_blocking=True)
        # Both OpenCLIP and OpenAI-CLIP expose encode_image
        emb = model.encode_image(images)
        if normalize:
            emb = emb / emb.norm(dim=-1, keepdim=True)
        feats.append(emb.detach().cpu().numpy().astype(np.float32))
        class_labels.append(y.numpy().astype(np.int64))
        client_ids.append(cids.numpy().astype(np.int64))
        indices.append(idxs.numpy().astype(np.int64))

    X = np.concatenate(feats, axis=0)
    y_cls = np.concatenate(class_labels, axis=0)
    y_usr = np.concatenate(client_ids, axis=0)
    idx = np.concatenate(indices, axis=0)
    return X, y_cls, y_usr, idx


def compute_separability_metrics(
    X: np.ndarray,
    y_class: np.ndarray,
    y_client: np.ndarray,
    sample_for_silhouette: int = 10000,
    random_state: int = 0,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    def safe_silhouette(features: np.ndarray, labels: np.ndarray) -> float:
        n = len(labels)
        if n < 50:
            return float("nan")
        if sample_for_silhouette and n > sample_for_silhouette:
            rng = RNG(random_state)
            sel = rng.choice(n, size=sample_for_silhouette, replace=False)
            features = features[sel]
            labels = labels[sel]
        try:
            return float(silhouette_score(features, labels, metric="cosine"))
        except Exception:
            return float("nan")

    # Silhouette scores
    metrics["silhouette_class_cosine"] = safe_silhouette(X, y_class)
    metrics["silhouette_client_cosine"] = safe_silhouette(X, y_client)

    # Calinski-Harabasz (higher is better)
    try:
        metrics["calinski_harabasz_class"] = float(calinski_harabasz_score(X, y_class))
    except Exception:
        metrics["calinski_harabasz_class"] = float("nan")
    try:
        metrics["calinski_harabasz_client"] = float(calinski_harabasz_score(X, y_client))
    except Exception:
        metrics["calinski_harabasz_client"] = float("nan")

    # Davies-Bouldin (lower is better)
    try:
        metrics["davies_bouldin_class"] = float(davies_bouldin_score(X, y_class))
    except Exception:
        metrics["davies_bouldin_class"] = float("nan")
    try:
        metrics["davies_bouldin_client"] = float(davies_bouldin_score(X, y_client))
    except Exception:
        metrics["davies_bouldin_client"] = float("nan")

    # Linear probe accuracies (class vs client)
    def linear_probe_acc(features: np.ndarray, labels: np.ndarray) -> float:
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=random_state, stratify=labels
        )
        clf = LogisticRegression(
            multi_class="auto",
            solver="lbfgs",
            max_iter=200,
            n_jobs=min(8, os.cpu_count() or 1),
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return float(accuracy_score(y_test, y_pred))

    try:
        metrics["linear_probe_class_acc"] = linear_probe_acc(X, y_class)
    except Exception:
        metrics["linear_probe_class_acc"] = float("nan")
    try:
        metrics["linear_probe_client_acc"] = linear_probe_acc(X, y_client)
    except Exception:
        metrics["linear_probe_client_acc"] = float("nan")

    return metrics


def plot_tsne(
    X: np.ndarray,
    labels: np.ndarray,
    title: str,
    save_path: str,
    num_points: int = 5000,
    random_state: int = 0,
) -> None:
    n = len(labels)
    if n == 0:
        return
    if n > num_points:
        rng = RNG(random_state)
        sel = rng.choice(n, size=num_points, replace=False)
        X = X[sel]
        labels = labels[sel]

    tsne = TSNE(
        n_components=2,
        perplexity=min(30, max(5, len(labels) // 100)),
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    )
    coords = tsne.fit_transform(X)

    plt.figure(figsize=(8, 7))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=4, alpha=0.7)
    plt.title(title)
    # One label per unique value
    handles, _ = scatter.legend_elements()
    uniq = sorted(np.unique(labels).tolist())
    plt.legend(handles, [str(u) for u in uniq], title="Label", loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze CLIP feature space separability on CIFAR-10 under non-IID (Dirichlet) partitions."
    )
    parser.add_argument("--num-clients", type=int, default=10, help="Number of clients/users")
    parser.add_argument("--alpha", type=float, default=0.3, help="Dirichlet concentration (smaller => more skew)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-B-32",
        help="CLIP model name (ViT-B-32, ViT-L-14, etc.)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-samples-total", type=int, default=20000, help="Cap on total samples to embed (<=0 means all)")
    parser.add_argument("--tsne-points", type=int, default=5000, help="Number of points used in t-SNE plots")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--save-dir", type=str, default="./results/clip_cifar10_non_iid")
    args = parser.parse_args()

    set_seed(args.seed)
    create_results_dir(args.save_dir)

    device = torch.device(args.device)
    model, preprocess, clip_meta = load_clip_model(args.model, device)

    # Dataset
    cifar_transform = transforms.Compose([transforms.ToTensor()])
    base_train = datasets.CIFAR10(root=args.data_root, train=True, download=True, transform=cifar_transform)
    targets = [int(y) for y in base_train.targets]

    # Non-IID partitions
    client_indices = create_dirichlet_partitions(
        y=targets,
        num_clients=args.num_clients,
        alpha=args.alpha,
        num_classes=10,
        min_size_per_client=100,
        seed=args.seed,
    )

    # Build unified subset with client ids and CLIP preprocessing
    indexed_samples = build_indexed_samples(
        targets=targets,
        client_indices=client_indices,
        max_samples_total=args.max_samples_total,
        seed=args.seed,
    )
    subset = IndexedSubset(base_dataset=datasets.CIFAR10(root=args.data_root, train=True, download=False), indexed_samples=indexed_samples, transform=preprocess)
    loader = data.DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(8, os.cpu_count() or 1),
        pin_memory=(device.type == "cuda"),
    )

    print(
        f"Encoding {human_size(len(subset))} samples with {args.model} on {device.type.upper()} | provider={clip_meta.get('provider')}"
    )
    X, y_class, y_client, idx = encode_features(model=model, loader=loader, device=device, normalize=True)
    print(f"Features: shape={X.shape}, dtype={X.dtype}")

    # Metrics
    metrics = compute_separability_metrics(X, y_class, y_client, sample_for_silhouette=10000, random_state=args.seed)

    # Expectation: class separability > client separability
    delta_silhouette = metrics.get("silhouette_class_cosine", math.nan) - metrics.get("silhouette_client_cosine", math.nan)
    metrics["silhouette_delta_class_minus_client"] = float(delta_silhouette)

    # Save metrics and embeddings
    npz_path = os.path.join(args.save_dir, "embeddings_and_labels.npz")
    np.savez_compressed(
        npz_path,
        features=X,
        class_labels=y_class,
        client_ids=y_client,
        indices=idx,
        meta=dict(
            model=args.model,
            provider=clip_meta.get("provider"),
            pretrained=clip_meta.get("pretrained"),
            alpha=args.alpha,
            num_clients=args.num_clients,
            seed=args.seed,
        ),
    )
    with open(os.path.join(args.save_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # t-SNE visualizations
    print("Running t-SNE (class coloring)...")
    plot_tsne(
        X,
        y_class,
        title="CLIP features: CIFAR-10 (non-IID) colored by class",
        save_path=os.path.join(args.save_dir, "tsne_classes.png"),
        num_points=args.tsne_points,
        random_state=args.seed,
    )

    print("Running t-SNE (client coloring)...")
    plot_tsne(
        X,
        y_client,
        title="CLIP features: CIFAR-10 (non-IID) colored by client/user",
        save_path=os.path.join(args.save_dir, "tsne_clients.png"),
        num_points=args.tsne_points,
        random_state=args.seed,
    )

    # Console summary
    print("\n=== Separability Metrics ===")
    for k in sorted(metrics.keys()):
        print(f"{k:36s}: {metrics[k]}")
    print(f"Saved embeddings to: {npz_path}")
    print(f"t-SNE plots saved to: {args.save_dir}")


if __name__ == "__main__":
    main()


