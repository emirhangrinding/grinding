## Project Structure

The code has been organized into the following modules:

### Core Modules

- **`utils.py`** - Utility functions including seed management, loss functions for disentanglement
- **`data.py`** - Data preparation, transforms, dataset generation, and data loaders
- **`models.py`** - Model definitions including ResNet-18 and multi-task learning model
- **`training.py`** - Training functions for the multi-task model
- **`evaluation.py`** - Evaluation functions including accuracy calculation and MIA

### Unlearning Methods

- **`dissolve.py`** - DISSOLVE unlearning implementation with FIM-based weight selection
- **`ssd.py`** - Selective Synaptic Dampening (SSD) unlearning implementation
- **`deepclean.py`** - DeepClean unlearning implementation with forget-sensitive weight fine-tuning
- **`retain_no_reset.py`** - Retain-No-Reset unlearning implementation without weight reset

### Additional Functionality

- **`baseline.py`** - Baseline training excluding one client for comparison
- **`tuning.py`** - Hyperparameter optimization for SSD using Optuna
- **`visualization.py`** - Plotting functions for training results
- **`main.py`** - CLI interface and main orchestration

## Usage

### Command Line Interface

The project provides a command-line interface through `main.py` with several subcommands:

#### 1. Training a Model

```bash
python main.py train \
    --dataset CIFAR10 \
    --setting non-iid \
    --num_clients 10 \
    --batch_size 256 \
    --num_epochs 200 \
    --model_path model.h5
```

#### 2. Unlearning with DISSOLVE

```bash
python main.py unlearn \
    --model_path model.h5 \
    --target_subset_id 0 \
    --gamma 0.1 \
    --beta 0.1 \
    --lr_unlearn 1e-3 \
    --epochs_unlearn 50 \
    --unlearning_type dissolve \
    --finetune_optimizer_type adam \
    --finetune_use_disentanglement_loss \
    --finetune_disentanglement_weight 0.5
```

#### 3. Unlearning with SSD

```bash
python main.py unlearn \
    --model_path model.h5 \
    --target_subset_id 0 \
    --unlearning_type ssd \
    --ssd_exponent 1.0 \
    --ssd_dampening_constant 0.5
```

#### 4. Unlearning with DeepClean

```bash
python main.py unlearn \
    --model_path model.h5 \
    --target_subset_id 0 \
    --gamma 0.1 \
    --lr_unlearn 1e-3 \
    --epochs_unlearn 50 \
    --unlearning_type deepclean \
    --finetune_optimizer_type sgd \
    --finetune_use_disentanglement_loss \
    --finetune_disentanglement_weight 1.0
```

#### 5. Unlearning with Retain-No-Reset

```bash
python main.py unlearn \
    --model_path model.h5 \
    --target_subset_id 0 \
    --gamma 0.1 \
    --beta 0.1 \
    --lr_unlearn 1e-3 \
    --epochs_unlearn 50 \
    --unlearning_type retain-no-reset \
    --finetune_task both
```

#### 6. Baseline Training (excluding one client)

```bash
python main.py baseline \
    --dataset CIFAR10 \
    --excluded_client_id 0 \
    --num_epochs 200 \
    --model_path baseline_model.h5
```

#### 7. Hyperparameter Tuning for SSD

```bash
python main.py tune_ssd \
    --model_path model.h5 \
    --target_subset_id 0 \
    --n_trials 25
```

### Fine-tuning Options for Unlearning Methods

The DISSOLVE and DeepClean unlearning methods support several fine-tuning parameters to customize the optimization process:

#### Optimizer Selection
- `--finetune_optimizer_type` or `finetune_optimizer_type`: Choose between "adam" (default) or "sgd"
  - Adam: Adaptive learning rate optimizer, generally converges faster
  - SGD: Stochastic Gradient Descent with momentum (0.9), often more stable

#### Disentanglement Loss Integration
- `--finetune_use_disentanglement_loss` or `finetune_use_disentanglement_loss`: Whether to include disentanglement loss during fine-tuning (default: False)
- `--finetune_disentanglement_weight` or `finetune_disentanglement_weight`: Weight for the disentanglement loss component (default: 1.0)

These parameters allow you to control the fine-tuning behavior to better suit your specific unlearning requirements. For example:
- Use SGD for more stable convergence when dealing with difficult unlearning scenarios
- Include disentanglement loss to maintain feature separation during fine-tuning
- Adjust the disentanglement weight to balance between task performance and feature disentanglement

### Programmatic Usage

You can also use the modules directly in your Python code:

#### Training a Model

```python
from training import learn
from models import MTL_Two_Heads_ResNet
from visualization import visualize_mtl_two_heads_results

trained_model, history, clients_labels = learn(
    dataset_name   = "CIFAR10",          # same dataset you will later unlearn from
    setting        = "non-iid",          # data-partition setting
    num_clients    = 10,                 # number of subsets/clients
    batch_size     = 128,                # mini-batch size
    num_epochs     = 200,                # training epochs (adjust as you like)
    lambda_1       = 1.0,                # weight for digit classification loss
    lambda_2       = 1.0,                # weight for subset identification loss
    lambda_dis     = 0.1,                # weight for disentanglement loss
    lambda_pull    = 1.0,                # weight for pull component in disentanglement
    lambda_push    = 1.0,                # weight for push component in disentanglement
    data_root      = "./data",           # where to download / look for CIFAR-10
    path           = "model_big.h5",     # where to save the trained weights
    model_class    = MTL_Two_Heads_ResNet,
    seed           = 42,                 # for reproducibility
    head_size      = "big"
)

# Visualize training results
visualize_mtl_two_heads_results(history)
```

#### Creating a Baseline Model

```python
from baseline import learn_baseline_excluding_client

model, history, metrics = learn_baseline_excluding_client(
    dataset_name="CIFAR10",
    setting="non-iid",
    num_clients=10,
    excluded_client_id=0,
    batch_size=128,
    num_epochs=200,
    lambda_1=1.0,
    lambda_2=1.0,
    lambda_dis=0.1,
    lambda_pull=1.0,
    lambda_push=1.0,
    data_root="./data",
    path="baseline_model_small.h5",
    seed=42,
    head_size="small",
)

print("\nBaseline Training Results:")
print(f"Final metrics: {metrics}")
```

#### Applying Unlearning

```python
from main import unlearn
from models import MTL_Two_Heads_ResNet

unlearned_model = unlearn(
    model_path                        = "model_200.h5",   # path to trained model
    target_subset_id                  = 0,                # client to forget
    unlearning                        = True,             # always True
    gamma                             = 0.1,              # forget-sensitivity threshold
    beta                              = 0.1,              # retain-sensitivity threshold
    lr_unlearn                        = 1e-3,             # learning rate for fine-tuning
    epochs_unlearn                    = 50,               # epochs for fine-tuning
    model_class                       = MTL_Two_Heads_ResNet,
    dataset_name                      = "CIFAR10",
    num_clients                       = 10,
    batch_size                        = 256,
    data_root                         = "./data",
    finetune_task                     = "both",           # "subset", "digit", or "both"
    fine_tune_heads                   = True,             # whether to fine-tune heads
    finetune_optimizer_type           = "adam",           # "adam" or "sgd" for fine-tuning
    finetune_use_disentanglement_loss = False,            # include disentanglement loss in fine-tuning
    finetune_disentanglement_weight   = 1.0,              # weight for disentanglement loss
    seed                              = 42,
    head_size                         = "medium",
    unlearning_type                   = "dissolve"        # "dissolve", "ssd", or "deepclean"
)
```

You can also call the unlearning methods directly:

```python
from dissolve import dissolve_unlearn_subset
from deepclean import deepclean_unlearn_subset
from retain_no_reset import retain_no_reset_unlearn_subset

# DISSOLVE with SGD optimizer and disentanglement loss
dissolve_model = dissolve_unlearn_subset(
    pretrained_model,
    retain_loader,
    forget_loader,
    target_subset_id=0,
    gamma=0.1,
    beta=0.1,
    lr_unlearn=1e-3,
    epochs_unlearn=50,
    device=device,
    finetune_task="both",
    finetune_optimizer_type="sgd",
    finetune_use_disentanglement_loss=True,
    finetune_disentanglement_weight=0.5
)

# DeepClean with Adam optimizer
deepclean_model = deepclean_unlearn_subset(
    pretrained_model,
    retain_loader,
    forget_loader,
    target_subset_id=0,
    gamma=0.1,
    lr_unlearn=1e-3,
    epochs_unlearn=50,
    device=device,
    finetune_task="subset",
    finetune_optimizer_type="adam",
    finetune_use_disentanglement_loss=False
)

# Retain-No-Reset with Adam optimizer
retain_no_reset_model = retain_no_reset_unlearn_subset(
    pretrained_model,
    retain_loader,
    forget_loader,
    target_subset_id=0,
    gamma=0.1,
    beta=0.1,
    lr_unlearn=1e-3,
    epochs_unlearn=50,
    device=device,
    finetune_task="both",
    finetune_optimizer_type="adam",
    finetune_use_disentanglement_loss=False
)
```

## Key Features

### Multi-Task Learning
- Joint training on digit classification and subset identification tasks
- Disentanglement loss to encourage feature separation
- Configurable head sizes (big, medium, small)

### DISSOLVE Unlearning
- Fisher Information Matrix (FIM) based weight selection
- Forget-sensitive weights are zeroed out
- Retain-sensitive weights are reset to random initialization
- Fine-tuning on retain set only

### SSD Unlearning
- Selective synaptic dampening based on parameter importance
- One-shot weight modification without fine-tuning
- Configurable dampening parameters

### DeepClean Unlearning
- Fisher Information Matrix (FIM) based weight selection
- Forget-sensitive weights are zeroed out then fine-tuned
- Fine-tuning only on forget-sensitive weights using retain set
- No modification of retain-sensitive weights

### Retain-No-Reset Unlearning
- Fisher Information Matrix (FIM) based weight selection
- Forget-sensitive weights are zeroed out (FIM_Df/FIM_Dr > gamma)
- Retain-sensitive weights are identified (FIM_Dr/FIM_Df > beta)
- Fine-tuning only on retain-sensitive weights using retain set
- No reset of retain-sensitive weights to random initialization

### Evaluation
- Comprehensive accuracy metrics for both tasks
- Membership Inference Attack (MIA) evaluation
- Test set performance tracking

## Dependencies

The project requires:
- PyTorch
- torchvision
- numpy
- matplotlib
- scikit-learn
- optuna (for hyperparameter tuning)

## File Overview

| File | Purpose | Key Functions/Classes |
|------|---------|----------------------|
| `utils.py` | Utilities and loss functions | `set_global_seed()`, `intra_y1_y2_disentanglement_loss()` |
| `data.py` | Data handling | `generate_subdatasets()`, `MultiTaskDataset` |
| `models.py` | Model definitions | `MTL_Two_Heads_ResNet`, `resnet18()` |
| `training.py` | Training logic | `train_mtl_two_heads()`, `learn()` |
| `evaluation.py` | Evaluation metrics | `calculate_*_accuracy()`, `get_membership_attack_prob_train_only()` |
| `dissolve.py` | DISSOLVE unlearning | `dissolve_unlearn_subset()` |
| `ssd.py` | SSD unlearning | `ssd_unlearn_subset()`, `ParameterPerturber` |
| `deepclean.py` | DeepClean unlearning | `deepclean_unlearn_subset()` |
| `retain_no_reset.py` | Retain-No-Reset unlearning | `retain_no_reset_unlearn_subset()` |
| `baseline.py` | Baseline training | `learn_baseline_excluding_client()` |
| `tuning.py` | Hyperparameter optimization | `optimise_ssd_hyperparams()` |
| `visualization.py` | Plotting | `visualize_mtl_two_heads_results()` |
| `main.py` | CLI interface | `main()`, `_build_cli_parser()` |

This modular structure makes the code much more maintainable, testable, and reusable compared to the original monolithic file. 