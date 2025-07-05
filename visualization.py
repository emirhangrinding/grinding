import matplotlib.pyplot as plt

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