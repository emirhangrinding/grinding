import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from evaluation import (
    calculate_digit_classification_accuracy,
    calculate_subset_identification_accuracy,
    calculate_overall_digit_classification_accuracy,
    calculate_overall_subset_identification_accuracy,
    get_membership_attack_prob_train_only
)

def calculate_fim_diagonal_subset(model, dataloader, device):
    """
    Calculate the diagonal of the Fisher Information Matrix for the subset identification task.
    """
    model.eval()
    fim_diag = {name: torch.zeros_like(param) for name, param in model.named_parameters() if param.requires_grad}
    num_batches_processed = 0  # count number of batches instead of individual samples

    if len(dataloader.dataset) == 0:
        print("Warning: dataloader is empty for FIM calculation. Returning zero FIM.")
        return fim_diag

    criterion = nn.CrossEntropyLoss()

    # SSD-style FIM: one backward pass per batch
    for inputs, _digit_labels, subset_labels in dataloader:
        inputs, subset_labels = inputs.to(device), subset_labels.to(device)

        # Forward + backward pass for the whole batch
        model.zero_grad()
        _digit_logits, subset_logits, _ = model(inputs)
        loss = criterion(subset_logits, subset_labels)  # mean reduction by default
        loss.backward()

        # Accumulate squared gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                fim_diag[name] += param.grad.data.pow(2)

        num_batches_processed += 1

    if num_batches_processed == 0:
        print("Warning: 0 batches processed for FIM calculation despite non-empty dataloader. Returning zero FIM.")
        return fim_diag

    # Average over batches
    for name in fim_diag:
        fim_diag[name] /= float(num_batches_processed)
    return fim_diag

def deepclean_unlearn_subset(
    pretrained_model, retain_loader, forget_loader,
    target_subset_id, gamma, lr_unlearn, epochs_unlearn, device,
    test_loader=None, finetune_task="subset", fine_tune_heads: bool = False,
    *, dataset_name: str = "CIFAR10", num_clients: int = 10, head_size: str = 'big'):
    """
    Unlearn subset classification using DeepClean method,
    but ONLY allow changes to resnet parameters (not heads).

    DeepClean approach:
    1. Zero out forget-sensitive weights (FIM_Df/FIM_Dr > gamma)
    2. Fine-tune only forget-sensitive weights on retain set
    3. Don't touch retain-sensitive weights

    finetune_task: 'subset', 'digit', or 'both'
    """
    mode_desc = "ResNet + Heads" if fine_tune_heads else "ResNet only"
    print(f"\n--- Starting DeepClean Unlearning for Subset ({mode_desc}) ---")   
    unlearned_model = copy.deepcopy(pretrained_model)
    unlearned_model.to(device)

    # 1. Calculate FIM diagonals for subset task
    print("Calculating FIM diagonal for D_f (forget set - target subset)...")
    fim_df = calculate_fim_diagonal_subset(unlearned_model, forget_loader, device)
    print("Calculating FIM diagonal for D_r (retain set - other subsets)...")
    fim_dr = calculate_fim_diagonal_subset(unlearned_model, retain_loader, device)

    # Determine adaptive thresholds for forget-sensitive weights
    # If `gamma` is < 1 it is interpreted as a fraction (e.g. 0.10 -> top 10%).
    # Otherwise it is treated as the absolute ratio threshold (back-compatibility).
    ratio_df_dr_dict = {}
    forget_ratio_list = []  # FIM_Df / FIM_Dr ratios
    for name, param in unlearned_model.named_parameters():
        if not name.startswith("resnet."):
            continue  # We only consider ResNet parameters for unlearning
        if name in fim_df and name in fim_dr:
            ratio_df_dr = fim_df[name] / (fim_dr[name] + 1e-9)
            ratio_df_dr_dict[name] = ratio_df_dr
            forget_ratio_list.append(ratio_df_dr.flatten())

    # Concatenate all ratios to compute quantile thresholds if needed
    all_forget_ratios = torch.cat(forget_ratio_list) if forget_ratio_list else torch.tensor([], device=device)

    if gamma < 1.0:
        # Interpret gamma as fraction for top-K selection
        k_forget = max(1, int(all_forget_ratios.numel() * gamma)) if all_forget_ratios.numel() > 0 else 0
        if k_forget == 0:
            forget_threshold = float("inf")
        else:
            topk_vals_forget, _ = torch.topk(all_forget_ratios, k=k_forget, largest=True)
            forget_threshold = topk_vals_forget.min().item()
    else:
        # Back-compatibility: treat gamma as absolute ratio threshold
        forget_threshold = gamma

    # 2. Identify forget-sensitive weights (W_f) only for resnet.* params
    forget_sensitivity_masks = {}
    W_f_param_names = []
    total_individual_weights = 0
    total_forget_sensitive_weights = 0

    if gamma < 1.0:
        print(f"\nIdentifying forget-sensitive individual weights in RESNET ONLY: top {gamma*100:.2f}% (threshold={forget_threshold:.4e}) based on FIM_Df/FIM_Dr ratio...")
    else:
        print(f"\nIdentifying forget-sensitive individual weights in RESNET ONLY with gamma threshold = {gamma}...")
    print("Layer Name     | Max I_Df | Max I_Dr | Min Ratio | Mean Ratio | Max Ratio | #Forget Sensitive | Total Weights | Contains Forget Sensitive?")
    print("----------------------------------------------------------------------------------------------------------------------------------------------")

    for name, param in unlearned_model.named_parameters():
        if not name.startswith("resnet."):
            # Only ResNet params may be unlearned
            param.requires_grad = False
            forget_sensitivity_masks[name] = torch.zeros_like(param.data, dtype=torch.bool)
            print(f"{name:<15} | N/A      | N/A      | N/A       | N/A        | N/A       | {0:18d} | {param.numel():13d} | False (Not ResNet)")
            continue

        total_individual_weights += param.numel()
        if name in fim_df and name in fim_dr:
            elementwise_ratio = fim_df[name] / (fim_dr[name] + 1e-9)
            current_mask = (elementwise_ratio >= forget_threshold)
            forget_sensitivity_masks[name] = current_mask

            num_forget_sensitive = current_mask.sum().item()
            total_forget_sensitive_weights += num_forget_sensitive

            contains_forget_sensitive = num_forget_sensitive > 0
            if contains_forget_sensitive:
                W_f_param_names.append(name)

            max_fim_df_val = fim_df[name].max().item() if fim_df[name].numel() > 0 else float('nan')
            max_fim_dr_val = fim_dr[name].max().item() if fim_dr[name].numel() > 0 else float('nan')
            min_ratio_val = elementwise_ratio.min().item() if elementwise_ratio.numel() > 0 else float('nan')
            mean_ratio_val = elementwise_ratio.mean().item() if elementwise_ratio.numel() > 0 else float('nan')
            max_ratio_val = elementwise_ratio.max().item() if elementwise_ratio.numel() > 0 else float('nan')

            print(f"{name:<15} | {max_fim_df_val:8.2e} | {max_fim_dr_val:8.2e} | {min_ratio_val:9.2f} | {mean_ratio_val:10.2f} | {max_ratio_val:9.2f} | {num_forget_sensitive:18d} | {param.numel():13d} | {contains_forget_sensitive}")
        else:
            forget_sensitivity_masks[name] = torch.zeros_like(param.data, dtype=torch.bool)
            print(f"{name:<15} | N/A      | N/A      | N/A       | N/A        | N/A       | {0:18d} | {param.numel():13d} | False (Not in FIM)")

    print("----------------------------------------------------------------------------------------------------------------------------------------------\n")

    if total_forget_sensitive_weights == 0:
        print("WARNING: No forget-sensitive weights were identified. Unlearning will not modify the model.")
        print(f"--- DeepClean Unlearning Finished ({mode_desc}) ---")
        return unlearned_model

    print(f"Total individual weights in RESNET: {total_individual_weights}")
    print(f"Total identified forget-sensitive weights (W_f): {total_forget_sensitive_weights}")
    print(f"{len(W_f_param_names)} parameter tensors contain forget-sensitive weights: {W_f_param_names}")

    # 3. Zero out forget-sensitive weights (W_f)
    print("Zeroing out forget-sensitive weights (W_f elements, RESNET ONLY)...")
    for name, param in unlearned_model.named_parameters():
        if not name.startswith("resnet."):
            continue
        if name in forget_sensitivity_masks:
            # Ensure mask is on the same device as the parameter tensor
            mask_wf = forget_sensitivity_masks[name].to(param.device)
            if mask_wf.any():
                with torch.no_grad():
                    param.data[mask_wf] = 0.0

    # 4. Set up fine-tuning parameters (forget-sensitive weights in ResNet and, optionally, heads)
    print("Setting up fine-tuning parameters...")
    params_to_fine_tune = []
    for name, param in unlearned_model.named_parameters():
        if name.startswith("resnet."):
            # ResNet parameters: honour W_f masks (only forget-sensitive weights will be fine-tuned)
            if name in forget_sensitivity_masks:
                mask_wf = forget_sensitivity_masks[name]
                if mask_wf.any():
                    param.requires_grad = True
                    params_to_fine_tune.append(param)
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = False
        elif fine_tune_heads and (name.startswith("digit_head.") or name.startswith("subset_head.")):
            # Optionally fine-tune the entire head parameters
            param.requires_grad = True
            params_to_fine_tune.append(param)
        else:
            param.requires_grad = False

    print("\nCalculating accuracies after weight modifications but before fine-tuning...")

    # Create a combined loader from retain and forget loaders for evaluation
    combined_dataset = []

    # Get all data from retain loader
    for batch_idx, (inputs, digit_labels, subset_labels) in enumerate(retain_loader):
        for i in range(inputs.size(0)):
            combined_dataset.append((inputs[i], digit_labels[i], subset_labels[i]))

    # Get all data from forget loader
    for batch_idx, (inputs, digit_labels, subset_labels) in enumerate(forget_loader):
        for i in range(inputs.size(0)):
            combined_dataset.append((inputs[i], digit_labels[i], subset_labels[i]))

    # Create a temporary DataLoader for evaluation
    class TempDataset(torch.utils.data.Dataset):
        def __init__(self, data_list):
            self.data = data_list

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    temp_dataset = TempDataset(combined_dataset)
    temp_loader = torch.utils.data.DataLoader(temp_dataset, batch_size=retain_loader.batch_size, shuffle=False)

    # Calculate accuracies after modifications
    target_accuracy_after_modifications, other_accuracy_after_modifications = \
        calculate_digit_classification_accuracy(unlearned_model, temp_loader, device, target_subset_id)
    subset_target_acc_after_modifications, subset_other_acc_after_modifications = \
        calculate_subset_identification_accuracy(unlearned_model, temp_loader, device, target_subset_id)

    print(f"Accuracy on target subset after weight modifications (before fine-tuning): {target_accuracy_after_modifications:.4f}")
    print(f"Accuracy on other subsets after weight modifications (before fine-tuning): {other_accuracy_after_modifications:.4f}")
    print(f"Subset ID accuracy on target subset after weight modifications (before fine-tuning): {subset_target_acc_after_modifications:.4f}")
    print(f"Subset ID accuracy on other subsets after weight modifications (before fine-tuning): {subset_other_acc_after_modifications:.4f}")

    # --- Test set evaluation after modifications ---
    if test_loader is not None:
        test_digit_acc = calculate_overall_digit_classification_accuracy(unlearned_model, test_loader, device)
        test_subset_acc = calculate_overall_subset_identification_accuracy(unlearned_model, test_loader, device)
        print(f"[TEST] Digit accuracy after weight modifications: {test_digit_acc:.4f}")
    
    # 5. Fine-tune (only W_f elements should be updated, only resnet)
    if not params_to_fine_tune:
        print("No forget-sensitive parameters to fine-tune. Skipping fine-tuning step.")
    else:
        print(f"Fine-tuning {total_forget_sensitive_weights} forget-sensitive weights (W_f) on D_r for {epochs_unlearn} epoch(s)...")
        optimizer_ft = optim.Adam(params_to_fine_tune, lr=lr_unlearn)
        criterion_ft = nn.CrossEntropyLoss()

        unlearned_model.train()
        for epoch in range(epochs_unlearn):
            running_loss_ft = 0.0
            total_samples_ft = 0

            for inputs, digit_labels, subset_labels in retain_loader:
                inputs = inputs.to(device)
                digit_labels = digit_labels.to(device)
                subset_labels = subset_labels.to(device)
                optimizer_ft.zero_grad()
                digit_logits, subset_logits, _ = unlearned_model(inputs)
                if finetune_task == "subset":
                    loss = criterion_ft(subset_logits, subset_labels)  # Fine-tuning based on subset classification
                elif finetune_task == "digit":
                    loss = criterion_ft(digit_logits, digit_labels)    # Fine-tuning based on digit classification
                elif finetune_task == "both":
                    loss = 0.6*criterion_ft(digit_logits, digit_labels) + 1.4*criterion_ft(subset_logits, subset_labels)  # Both losses
                else:
                    raise ValueError(f"finetune_task must be 'subset', 'digit', or 'both', got {finetune_task}")

                loss.backward()

                # Zero-out gradients for non-forget-sensitive elements within the resnet tensors being optimized
                for name, param in unlearned_model.named_parameters():
                    if name.startswith("resnet.") and param.grad is not None and name in forget_sensitivity_masks:
                        # Ensure mask is on the same device as the parameter tensor
                        mask_non_wf_elements = ~forget_sensitivity_masks[name].to(param.device)
                        param.grad.data[mask_non_wf_elements] = 0.0

                optimizer_ft.step()
                running_loss_ft += loss.item() * inputs.size(0)
                total_samples_ft += inputs.size(0)

            if total_samples_ft > 0:
                epoch_loss_ft = running_loss_ft / total_samples_ft
                print(f"Fine-tuning Epoch {epoch+1}/{epochs_unlearn}, Loss on D_r: {epoch_loss_ft:.4f}")
            else:
                print(f"Fine-tuning Epoch {epoch+1}/{epochs_unlearn}, No samples in D_r for fine-tuning.")

            if temp_loader: # Ensure temp_loader was successfully created
                unlearned_model.eval() # Set model to evaluation mode for accuracy calculation
                # Using temp_loader which contains both retain and forget data for a comprehensive evaluation
                unlearned_target_accuracy, unlearned_other_accuracy = \
                    calculate_digit_classification_accuracy(unlearned_model, temp_loader, device, target_subset_id)
                unlearned_subset_target_acc, unlearned_subset_other_acc = \
                    calculate_subset_identification_accuracy(unlearned_model, temp_loader, device, target_subset_id)
                print(f"  Unlearned accuracy on target subset after finetuning epoch {epoch+1}: {unlearned_target_accuracy:.4f}")
                print(f"  Unlearned accuracy on other subsets after finetuning epoch {epoch+1}: {unlearned_other_accuracy:.4f}")
                print(f"  Unlearned subset ID accuracy on target subset after finetuning epoch {epoch+1}: {unlearned_subset_target_acc:.4f}")
                print(f"  Unlearned subset ID accuracy on other subsets after finetuning epoch {epoch+1}: {unlearned_subset_other_acc:.4f}")
                # --- Test set evaluation after each epoch ---
                if test_loader is not None:
                    test_digit_acc = calculate_overall_digit_classification_accuracy(unlearned_model, test_loader, device)
                    test_subset_acc = calculate_overall_subset_identification_accuracy(unlearned_model, test_loader, device)
                    print(f"    [TEST] Digit accuracy: {test_digit_acc:.4f}")
            else:
                print(f"  Skipping accuracy calculation after finetuning epoch {epoch+1} as temp_loader is not available.")
            
            unlearned_model.train()  # Set back to training mode for next epoch

        unlearned_model.eval()

    print(f"--- DeepClean Unlearning Finished ({mode_desc}) ---")

    # DeepClean unlearning finished for the chosen parameter subset.
    return unlearned_model 