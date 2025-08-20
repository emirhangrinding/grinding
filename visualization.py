import matplotlib.pyplot as plt
import numpy as np

def visualize_mtl_two_heads_results(history):
    """
    Visualize the results of the Multi-Task Learning experiment
    """
    plt.figure(figsize=(15, 5))

    # Plot losses
    plt.subplot(1, 3, 1)
    plt.plot(history['train_digit_loss'], label='Train Digit Loss')
    plt.plot(history['test_digit_loss'], label='Test Digit Loss')
    plt.plot(history['train_subset_loss'], label='Train Subset Loss')
    plt.plot(history['train_loss_dis'], label='Train Disentanglement Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Task Losses')

    # Plot digit classification accuracy
    plt.subplot(1, 3, 2)
    plt.plot(history['train_digit_acc'], label='Train Digit Acc')
    plt.plot(history['test_digit_acc'], label='Test Digit Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Digit Classification Accuracy')

    # Plot subset identification accuracy
    plt.subplot(1, 3, 3)
    plt.plot(history['train_subset_acc'], label='Train Subset Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Subset Identification Accuracy (Training Only)')

    plt.tight_layout()
    plt.savefig('mtl_two_heads_results.png')
    plt.show() 


def plot_client_class_heatmap(clients_labels, num_classes=10, title='Client-Class Distribution (Non-IID Dirichlet)', save_path='client_class_heatmap.png', dpi=600):
    """
    Given a mapping of client_id -> {class_label: count},
    print per-client sample counts and save a heatmap figure for paper-ready usage.

    Args:
        clients_labels: dict like {"client1": {0: n0, 1: n1, ...}, "client2": {...}, ...}
        num_classes: total number of classes in the dataset
        title: figure title
        save_path: file path to save the figure
        dpi: image DPI for saving (use >= 300 for paper quality)
    """
    # Ensure deterministic client order (client1, client2, ...)
    client_keys = sorted(clients_labels.keys(), key=lambda k: int(k.replace('client', '')))

    # Build matrix: rows=clients, cols=classes
    matrix = []
    client_totals = []
    for ck in client_keys:
        counts_for_client = clients_labels.get(ck, {})
        row = [int(counts_for_client.get(c, 0)) for c in range(num_classes)]
        matrix.append(row)
        client_totals.append(int(sum(row)))

    matrix = np.array(matrix, dtype=np.int64)

    # Print per-client sample counts
    print('Per-client sample counts:')
    for ck, total in zip(client_keys, client_totals):
        print(f"{ck}: {total}")

    # Create heatmap
    fig, ax = plt.subplots(figsize=(max(8, num_classes * 0.6), max(5, len(client_keys) * 0.4)))
    im = ax.imshow(matrix, aspect='auto', cmap='viridis')

    # Axis labels and ticks
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Class label', fontsize=12)
    ax.set_ylabel('Client', fontsize=12)
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(len(client_keys)))
    ax.set_yticklabels(client_keys)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Number of samples', rotation=90)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def visualize_dirichlet_split_and_heatmap(dataset_name='CIFAR10', setting='non-iid', num_clients=10, data_root='./data', num_classes=10, save_path='client_class_heatmap.png', dpi=600):
    """
    Utility to generate the non-iid Dirichlet split using the project's data utilities,
    print per-client counts, and save a heatmap.

    Note: The Dirichlet alpha is configured inside generate_subdatasets for the 'non-iid' setting.
    """
    # Lazy import to avoid circular imports at module load time
    from data import generate_subdatasets

    clients_data, clients_labels, _ = generate_subdatasets(
        dataset_name=dataset_name,
        setting=setting,
        num_clients=num_clients,
        data_root=data_root
    )

    # clients_labels is already in the desired format
    plot_client_class_heatmap(
        clients_labels,
        num_classes=num_classes,
        title=f'{dataset_name} {setting.upper()} Dirichlet: Client-Class Distribution',
        save_path=save_path,
        dpi=dpi,
    )