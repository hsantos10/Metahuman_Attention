# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 18:30:06 2025

@author: David
"""
# train.py
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import copy

def train_and_evaluate(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    device,
    loss_weights=None,
    early_stopping_patience=None,
    early_stopping_min_delta=0.0,
    diagnostics=False,
    target_feature_names=None,
    on_epoch_end=None,
):
    """
    Trains and evaluates the model, incorporating masking for padded sequences.

    Args:
        model: The PyTorch model to train.
        train_loader: DataLoader for the training set. Yields inputs, targets, and input_masks.
        val_loader: DataLoader for the validation set. Yields inputs, targets, and input_masks.
        criterion: The loss function. Should return element-wise losses (shape: batch, seq_len, num_features)
                   for proper masking and per-feature weighting. Use reduction='none' for standard losses.
                   Custom losses (e.g., NormalizedPeakWeightedLoss) should return element-wise losses.
        optimizer: The optimization algorithm.
        scheduler: The learning rate scheduler.
        num_epochs (int): The number of epochs to train for.
        device: The device to train on ('cuda' or 'cpu').
        loss_weights (list, optional): A list of weights for each output feature. 
                                       Only applied if using standard loss functions (not custom losses
                                       that handle their own weighting like NormalizedPeakWeightedLoss).
                                       Defaults to None.
        early_stopping_patience (int, optional): Stop training early when validation
                           loss does not improve for this many epochs.
        early_stopping_min_delta (float, optional): Minimum drop in validation loss
                           required to count as improvement.

    Returns:
        tuple: Containing lists of training/validation losses, MAEs, and R2 scores.
    """
    def _safe_r2(y_true, y_pred):
        if len(y_true) > 1 and np.var(y_true) > 1e-6:
            return float(r2_score(y_true, y_pred))
        if len(y_true) > 1 and np.allclose(y_true, y_pred):
            return 1.0
        return 0.0

    def _feature_name(idx):
        if target_feature_names is not None and idx < len(target_feature_names):
            return str(target_feature_names[idx])
        return f"feature_{idx}"

    model.to(device)
    train_losses, val_losses = [], []
    train_maes, val_maes = [], []
    train_r2s, val_r2s = [], []
    
    best_val_mae = float('inf')  # Initialize with a very high value
    best_val_loss = float('inf')
    best_model_state_dict = None # Will store the state_dict of the best model
    best_epoch = -1              # Will store the epoch number of the best model
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        # --- Training phase ---
        model.train()
        batch_train_loss_sum = 0
        batch_train_mae_sum = 0
        total_train_elements = 0 # For averaging masked loss/mae

        # Initialize lists for storing epoch-level data for R2 calculation
        epoch_train_actuals_list = []
        epoch_train_predictions_list = []
        epoch_train_masks_list = [] # To store masks corresponding to targets

        # DataLoader yields inputs, metadata, targets, and input_masks_from_loader
        for i, (inputs, metadata, targets, input_masks_from_loader) in enumerate(train_loader): # Unpack metadata
            inputs = inputs.to(device)
            metadata = metadata.to(device) # Move metadata to device for model input
            targets = targets.to(device)
            input_masks_dev = input_masks_from_loader.to(device) # Mask for model input

            # Print shape for the first batch of the first epoch
            if epoch == 0 and i == 0:
                print(f"DEBUG: Shape of 'inputs' tensor going into model: {inputs.shape}")
                print(f"DEBUG: Shape of 'metadata' tensor going into model: {metadata.shape}")
                print(f"DEBUG: Shape of 'targets' tensor for loss calculation: {targets.shape}")
                print(f"DEBUG: Shape of 'input_masks_dev' tensor: {input_masks_dev.shape}")
            
            optimizer.zero_grad()
            
            outputs = model(inputs, metadata, input_mask=input_masks_dev) # (batch, seq_len, num_features)
            
            if len(outputs.shape) == 2 and len(targets.shape) == 3:
                outputs = outputs.unsqueeze(1)

            if outputs.shape[1] != targets.shape[1]:
                outputs = outputs.expand(-1, targets.shape[1], -1)

            raw_loss = criterion(outputs, targets)

            if loss_weights is not None:
                loss_weights_tensor = torch.tensor(loss_weights, device=device).unsqueeze(0).unsqueeze(0)
                raw_loss = raw_loss * loss_weights_tensor

            target_mask_for_loss = input_masks_dev.unsqueeze(-1).expand_as(targets)

            masked_loss_elements = raw_loss * target_mask_for_loss
            loss = masked_loss_elements.sum() / target_mask_for_loss.sum().clamp(min=1e-9)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            batch_train_loss_sum += masked_loss_elements.sum().item()
            
            abs_error = torch.abs(outputs - targets)
            masked_abs_error_elements = abs_error * target_mask_for_loss
            batch_train_mae_sum += masked_abs_error_elements.sum().item()
            
            total_train_elements += target_mask_for_loss.sum().item()

            epoch_train_actuals_list.append(targets.cpu().numpy())
            epoch_train_predictions_list.append(outputs.detach().cpu().numpy())
            epoch_train_masks_list.append(target_mask_for_loss.cpu().numpy())
            
        avg_train_loss = batch_train_loss_sum / total_train_elements if total_train_elements > 0 else 0
        avg_train_mae = batch_train_mae_sum / total_train_elements if total_train_elements > 0 else 0
        train_losses.append(avg_train_loss)
        train_maes.append(avg_train_mae)

        train_actuals_epoch = np.concatenate(epoch_train_actuals_list, axis=0)
        train_predictions_epoch = np.concatenate(epoch_train_predictions_list, axis=0)
        train_masks_epoch = np.concatenate(epoch_train_masks_list, axis=0)

        num_features = train_actuals_epoch.shape[-1]
        train_r2_scores_per_feature = []
        for feat_idx in range(num_features):
            actuals_feat = train_actuals_epoch[:, :, feat_idx][train_masks_epoch[:, :, feat_idx] > 0]
            preds_feat = train_predictions_epoch[:, :, feat_idx][train_masks_epoch[:, :, feat_idx] > 0]
            
            if hasattr(actuals_feat, 'cpu') and hasattr(actuals_feat, 'numpy'):
                actuals_feat_np = actuals_feat.cpu().numpy()
            elif hasattr(actuals_feat, 'numpy'):
                actuals_feat_np = actuals_feat.numpy()
            else:
                actuals_feat_np = np.asarray(actuals_feat)

            if hasattr(preds_feat, 'cpu') and hasattr(preds_feat, 'numpy'):
                preds_feat_np = preds_feat.cpu().numpy()
            elif hasattr(preds_feat, 'numpy'):
                preds_feat_np = preds_feat.numpy()
            else:
                preds_feat_np = np.asarray(preds_feat)

            epoch_num = epoch + 1 if 'epoch' in locals() else 'N/A'
            feature_index = feat_idx if 'feat_idx' in locals() else 'N/A'

            if np.isnan(actuals_feat_np).any() or np.isinf(actuals_feat_np).any():
                print(f"DEBUG_TRAIN [Epoch {epoch_num}, Feat. Index {feature_index}]: actuals_feat_np contains NaN/Inf BEFORE r2_score. Shape: {actuals_feat_np.shape}")
            if np.isnan(preds_feat_np).any() or np.isinf(preds_feat_np).any():
                print(f"DEBUG_TRAIN [Epoch {epoch_num}, Feat. Index {feature_index}]: preds_feat_np contains NaN/Inf BEFORE r2_score. Shape: {preds_feat_np.shape}")

            actuals_feat_safe = np.nan_to_num(actuals_feat_np, nan=0.0, posinf=0.0, neginf=0.0)
            preds_feat_safe = np.nan_to_num(preds_feat_np, nan=0.0, posinf=0.0, neginf=0.0)

            if len(actuals_feat_safe) > 1 and np.var(actuals_feat_safe) > 1e-6 :
                current_r2_score = r2_score(actuals_feat_safe, preds_feat_safe)
            elif len(actuals_feat_safe) > 1 and np.allclose(actuals_feat_safe, preds_feat_safe):
                 current_r2_score = 1.0
            else:
                 current_r2_score = 0.0

            train_r2_scores_per_feature.append(current_r2_score)
        
        avg_train_r2 = np.mean(train_r2_scores_per_feature) if train_r2_scores_per_feature else 0.0
        train_r2s.append(avg_train_r2)

        if val_loader is not None:
            model.eval()
            batch_val_loss_sum = 0
            batch_val_mae_sum = 0
            total_val_elements = 0

            epoch_val_actuals_list = []
            epoch_val_predictions_list = []
            epoch_val_masks_list = []

            with torch.no_grad():
                for inputs, metadata, targets, input_masks_from_loader in val_loader:
                    inputs = inputs.to(device)
                    metadata = metadata.to(device)
                    targets = targets.to(device)
                    input_masks_dev = input_masks_from_loader.to(device)

                    outputs = model(inputs, metadata, input_mask=input_masks_dev)

                    if len(outputs.shape) == 2 and len(targets.shape) == 3:
                        outputs = outputs.unsqueeze(1)

                    if outputs.shape[1] != targets.shape[1]:
                        outputs = outputs.expand(-1, targets.shape[1], -1)

                    raw_loss = criterion(outputs, targets)
                    target_mask_for_loss = input_masks_dev.unsqueeze(-1).expand_as(targets)
                    masked_loss_elements = raw_loss * target_mask_for_loss

                    batch_val_loss_sum += masked_loss_elements.sum().item()

                    abs_error = torch.abs(outputs - targets)
                    masked_abs_error_elements = abs_error * target_mask_for_loss
                    batch_val_mae_sum += masked_abs_error_elements.sum().item()

                    total_val_elements += target_mask_for_loss.sum().item()

                    epoch_val_actuals_list.append(targets.cpu().numpy())
                    epoch_val_predictions_list.append(outputs.cpu().numpy())
                    epoch_val_masks_list.append(target_mask_for_loss.cpu().numpy())

            avg_val_loss = batch_val_loss_sum / total_val_elements if total_val_elements > 0 else 0
            avg_val_mae = batch_val_mae_sum / total_val_elements if total_val_elements > 0 else 0
            val_losses.append(avg_val_loss)
            val_maes.append(avg_val_mae)

            if avg_val_mae < best_val_mae:
                best_val_mae = avg_val_mae
                print(f"Epoch {epoch + 1}: New best validation MAE: {best_val_mae:.4f}")

            if avg_val_loss < (best_val_loss - early_stopping_min_delta):
                best_val_loss = avg_val_loss
                best_model_state_dict = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1
                early_stopping_counter = 0
                print(f"Epoch {best_epoch}: New best validation loss: {best_val_loss:.4f}")
            elif early_stopping_patience is not None:
                early_stopping_counter += 1

            val_actuals_epoch = np.concatenate(epoch_val_actuals_list, axis=0)
            val_predictions_epoch = np.concatenate(epoch_val_predictions_list, axis=0)
            val_masks_epoch = np.concatenate(epoch_val_masks_list, axis=0)

            val_r2_scores_per_feature = []
            baseline_r2_scores_per_feature = []
            for feat_idx in range(num_features):
                if feat_idx >= val_actuals_epoch.shape[-1] or feat_idx >= val_masks_epoch.shape[-1] or feat_idx >= val_predictions_epoch.shape[-1]:
                    val_r2_scores_per_feature.append(0.0)
                    baseline_r2_scores_per_feature.append(0.0)
                    continue
                actuals_feat = val_actuals_epoch[:, :, feat_idx][val_masks_epoch[:, :, feat_idx] > 0]
                preds_feat = val_predictions_epoch[:, :, feat_idx][val_masks_epoch[:, :, feat_idx] > 0]

                if hasattr(actuals_feat, 'cpu') and hasattr(actuals_feat, 'numpy'):
                    actuals_feat_np = actuals_feat.cpu().numpy()
                elif hasattr(actuals_feat, 'numpy'):
                    actuals_feat_np = actuals_feat.numpy()
                else:
                    actuals_feat_np = np.asarray(actuals_feat)

                if hasattr(preds_feat, 'cpu') and hasattr(preds_feat, 'numpy'):
                    preds_feat_np = preds_feat.cpu().numpy()
                elif hasattr(preds_feat, 'numpy'):
                    preds_feat_np = preds_feat.numpy()
                else:
                    preds_feat_np = np.asarray(preds_feat)

                epoch_num = epoch + 1 if 'epoch' in locals() else 'N/A'
                feature_index = feat_idx if 'feat_idx' in locals() else 'N/A'

                if np.isnan(actuals_feat_np).any() or np.isinf(actuals_feat_np).any():
                    print(f"DEBUG_TRAIN [Epoch {epoch_num}, Feat. Index {feature_index} VAL]: actuals_feat_np contains NaN/Inf BEFORE r2_score. Shape: {actuals_feat_np.shape}")
                if np.isnan(preds_feat_np).any() or np.isinf(preds_feat_np).any():
                    print(f"DEBUG_TRAIN [Epoch {epoch_num}, Feat. Index {feature_index} VAL]: preds_feat_np contains NaN/Inf BEFORE r2_score. Shape: {preds_feat_np.shape}")

                actuals_feat_safe = np.nan_to_num(actuals_feat_np, nan=0.0, posinf=0.0, neginf=0.0)
                preds_feat_safe = np.nan_to_num(preds_feat_np, nan=0.0, posinf=0.0, neginf=0.0)

                current_r2_score = _safe_r2(actuals_feat_safe, preds_feat_safe)

                baseline_value = float(np.mean(actuals_feat_safe)) if len(actuals_feat_safe) > 0 else 0.0
                baseline_pred = np.full_like(actuals_feat_safe, baseline_value)
                baseline_r2 = _safe_r2(actuals_feat_safe, baseline_pred)

                val_r2_scores_per_feature.append(current_r2_score)
                baseline_r2_scores_per_feature.append(baseline_r2)

                should_log_diag = diagnostics and (epoch == 0 or (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs)
                if should_log_diag:
                    pred_std = float(np.std(preds_feat_safe)) if len(preds_feat_safe) > 0 else 0.0
                    target_std = float(np.std(actuals_feat_safe)) if len(actuals_feat_safe) > 0 else 0.0
                    pred_mean = float(np.mean(preds_feat_safe)) if len(preds_feat_safe) > 0 else 0.0
                    target_mean = float(np.mean(actuals_feat_safe)) if len(actuals_feat_safe) > 0 else 0.0
                    corr = np.corrcoef(actuals_feat_safe, preds_feat_safe)[0, 1] if len(actuals_feat_safe) > 1 else np.nan
                    if not np.isfinite(corr):
                        corr = np.nan

                    const_flag = " [PRED_NEAR_CONSTANT]" if pred_std < 1e-6 else ""
                    print(
                        f"VAL_DIAG [Epoch {epoch + 1}] {_feature_name(feat_idx)}: "
                        f"target_mean={target_mean:.4f}, target_std={target_std:.4f}, "
                        f"pred_mean={pred_mean:.4f}, pred_std={pred_std:.4f}, "
                        f"corr={corr:.4f}, model_r2={current_r2_score:.4f}, baseline_r2={baseline_r2:.4f}{const_flag}"
                    )

            avg_val_r2 = np.mean(val_r2_scores_per_feature) if val_r2_scores_per_feature else 0.0
            avg_baseline_val_r2 = np.mean(baseline_r2_scores_per_feature) if baseline_r2_scores_per_feature else 0.0
            val_r2s.append(avg_val_r2)

            # Optional per-epoch hook (used by HPO for Optuna pruning). The
            # callback may raise (e.g. optuna.TrialPruned) to stop training early.
            if on_epoch_end is not None:
                on_epoch_end(epoch, avg_val_r2)

            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train MAE: {avg_train_mae:.4f}, Val MAE: {avg_val_mae:.4f}, Train R2: {avg_train_r2:.4f}, Val R2: {avg_val_r2:.4f}')
            if diagnostics:
                print(
                    f"VAL_DIAG [Epoch {epoch + 1}] avg_model_r2={avg_val_r2:.4f}, "
                    f"avg_mean_baseline_r2={avg_baseline_val_r2:.4f}, "
                    f"r2_gap={avg_val_r2 - avg_baseline_val_r2:.4f}"
                )

            scheduler.step(avg_val_loss)

            if early_stopping_patience is not None and early_stopping_counter >= early_stopping_patience:
                print(
                    f"Early stopping triggered at epoch {epoch + 1}. "
                    f"No validation-loss improvement for {early_stopping_patience} epochs."
                )
                break
        else:
            val_losses.append(float('nan'))
            val_maes.append(float('nan'))
            val_r2s.append(float('nan'))
            best_model_state_dict = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train MAE: {avg_train_mae:.4f}, Train R2: {avg_train_r2:.4f} (No validation)')

    if best_model_state_dict is None:
        print("Warning: best_model_state_dict is None after training loop. Saving current model state.")
        best_model_state_dict = copy.deepcopy(model.state_dict())
        best_epoch = num_epochs if num_epochs > 0 else 0

    if best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)
        val_mae_str = f"{best_val_mae:.4f}" if val_loader and best_val_mae != float('inf') else "N/A"
        print(f"Loaded best model state from epoch {best_epoch} (Val MAE: {val_mae_str}) into model.")
    else:
        print("Warning: No best model state was found or loaded.")


    return train_losses, val_losses, train_maes, val_maes, train_r2s, val_r2s

def validate_model(model, data_loader, device):
    model.eval()
    predictions_list, actuals_list, masks_list = [], [], []
    with torch.no_grad():
        for inputs, metadata, targets, input_masks_from_loader in data_loader:
            inputs, metadata, targets, input_masks_dev = (
                inputs.to(device), metadata.to(device), targets.to(device), input_masks_from_loader.to(device)
            )
            outputs = model(inputs, metadata, input_mask=input_masks_dev)
            if len(outputs.shape) == 2:
                outputs = outputs.unsqueeze(1).expand(-1, targets.shape[1], -1)
            predictions_list.append(outputs.cpu().numpy())
            actuals_list.append(targets.cpu().numpy())
            masks_list.append(input_masks_from_loader.cpu().numpy())
    return np.concatenate(predictions_list, axis=0), np.concatenate(actuals_list, axis=0), np.concatenate(masks_list, axis=0)

def analyze_results(predictions, actuals, feature_names, masks=None):
    num_samples, seq_len, num_features = actuals.shape
    
    if masks is not None:
        # Expand masks to match features: (N, L) -> (N, L, F)
        masks_expanded = np.repeat(masks[:, :, np.newaxis], num_features, axis=2)
        valid_indices = masks_expanded > 0

        # We can't simply reshape if we want to ignore padding,
        # so we calculate metrics per feature using masking.
        mae, mse, r2_per_feature, baseline_r2_per_feature = [], [], [], []

        for i in range(num_features):
            # Select valid elements for this feature across all samples and timesteps
            valid_mask = masks > 0
            pred_f = predictions[:, :, i][valid_mask]
            act_f = actuals[:, :, i][valid_mask]

            if len(act_f) == 0:
                mae.append(0.0)
                mse.append(0.0)
                r2_per_feature.append(0.0)
                baseline_r2_per_feature.append(0.0)
                continue

            # MAE & MSE
            mae.append(np.mean(np.abs(pred_f - act_f)))
            mse_val = np.mean((pred_f - act_f) ** 2)
            mse.append(mse_val)

            # R2
            ss_res = np.sum((act_f - pred_f) ** 2)
            ss_tot = np.sum((act_f - np.mean(act_f)) ** 2)

            if ss_tot == 0:
                r2_val = 1.0 if ss_res == 0 else 0.0
            else:
                r2_val = 1 - (ss_res / ss_tot)
            r2_per_feature.append(r2_val)

            baseline_pred = np.full_like(act_f, np.mean(act_f))
            baseline_ss_res = np.sum((act_f - baseline_pred) ** 2)
            baseline_r2 = 1 - (baseline_ss_res / ss_tot) if ss_tot != 0 else 0.0
            baseline_r2_per_feature.append(baseline_r2)

        mae = np.array(mae)
        mse = np.array(mse)
        rmse = np.sqrt(mse)
        
    else:
        # Fallback to legacy behavior (includes padding)
        predictions_flat = predictions.reshape(-1, num_features)
        actuals_flat = actuals.reshape(-1, num_features)

        mae = np.mean(np.abs(predictions_flat - actuals_flat), axis=0)
        mse = np.mean((predictions_flat - actuals_flat) ** 2, axis=0)
        rmse = np.sqrt(mse)

        r2_per_feature = []
        baseline_r2_per_feature = []
        for i in range(num_features):
            ss_res = np.sum((actuals_flat[:, i] - predictions_flat[:, i]) ** 2)
            ss_tot = np.sum((actuals_flat[:, i] - np.mean(actuals_flat[:, i])) ** 2)

            if ss_tot == 0:
                r2_val = 1.0 if ss_res == 0 else 0.0
            else:
                r2_val = 1 - (ss_res / ss_tot)
            r2_per_feature.append(r2_val)

            baseline_pred = np.full_like(actuals_flat[:, i], np.mean(actuals_flat[:, i]))
            baseline_ss_res = np.sum((actuals_flat[:, i] - baseline_pred) ** 2)
            baseline_r2 = 1 - (baseline_ss_res / ss_tot) if ss_tot != 0 else 0.0
            baseline_r2_per_feature.append(baseline_r2)

    results = pd.DataFrame({
        'Feature': feature_names,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2_per_feature,
        'Baseline_R2_MeanPredictor': baseline_r2_per_feature,
        'R2_Gap_vs_Baseline': np.array(r2_per_feature) - np.array(baseline_r2_per_feature),
    })
    print("\nAnalysis Results (per feature, across all time steps):")
    print(results)
    return results
