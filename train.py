# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 18:30:06 2025

@author: David
"""
# train.py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """
    Trains and evaluates the model, incorporating masking for padded sequences.

    Args:
        model: The PyTorch model to train.
        train_loader: DataLoader for the training set. Yields inputs, targets, and input_masks.
        val_loader: DataLoader for the validation set. Yields inputs, targets, and input_masks.
        criterion: The loss function (e.g., nn.MSELoss(reduction='none')).
        optimizer: The optimization algorithm.
        num_epochs (int): The number of epochs to train for.
        device: The device to train on ('cuda' or 'cpu').

    Returns:
        tuple: Containing lists of training/validation losses, MAEs, and R2 scores.
    """
    model.to(device)
    train_losses, val_losses = [], []
    train_maes, val_maes = [], []
    train_r2s, val_r2s = [], []

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

        # DataLoader yields inputs, targets, and input_masks_from_loader
        for inputs, targets, input_masks_from_loader in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            input_masks_dev = input_masks_from_loader.to(device) # Mask for model input
            
            optimizer.zero_grad()
            
            # Pass input_masks_dev to the model's forward method
            outputs = model(inputs, input_mask=input_masks_dev) # (batch, seq_len, num_features)
            
            # Loss Calculation (Masked)
            # criterion should have reduction='none', raw_loss is (batch, seq_len, num_features)
            raw_loss = criterion(outputs, targets) 
            
            # Create a target_mask from input_masks_dev (or a separate target_mask if available)
            # This mask is applied to the loss and metrics for the target sequences.
            # Assuming target padding mirrors input padding for now.
            target_mask_for_loss = input_masks_dev.unsqueeze(-1).expand_as(targets) # (B, S, 1) -> (B, S, F)

            masked_loss_elements = raw_loss * target_mask_for_loss
            # Average loss only over non-padded elements
            loss = masked_loss_elements.sum() / target_mask_for_loss.sum().clamp(min=1e-9) # Avoid division by zero

            loss.backward()
            optimizer.step()
            
            batch_train_loss_sum += masked_loss_elements.sum().item() # Sum of loss over non-padded
            
            # MAE Calculation (Masked)
            abs_error = torch.abs(outputs - targets)
            masked_abs_error_elements = abs_error * target_mask_for_loss
            batch_train_mae_sum += masked_abs_error_elements.sum().item()
            
            total_train_elements += target_mask_for_loss.sum().item()

            # Store full sequences and masks for epoch-level R2 calculation
            epoch_train_actuals_list.append(targets.cpu().numpy())
            epoch_train_predictions_list.append(outputs.detach().cpu().numpy())
            epoch_train_masks_list.append(target_mask_for_loss.cpu().numpy()) # Store the mask used for loss/metrics
            
        avg_train_loss = batch_train_loss_sum / total_train_elements if total_train_elements > 0 else 0
        avg_train_mae = batch_train_mae_sum / total_train_elements if total_train_elements > 0 else 0
        train_losses.append(avg_train_loss)
        train_maes.append(avg_train_mae)

        # Concatenate all batch results for the epoch
        train_actuals_epoch = np.concatenate(epoch_train_actuals_list, axis=0)
        train_predictions_epoch = np.concatenate(epoch_train_predictions_list, axis=0)
        train_masks_epoch = np.concatenate(epoch_train_masks_list, axis=0) # Shape: (N_samples, Seq_len, Num_features)

        # R2 Calculation (Masked) for Training Data
        num_features = train_actuals_epoch.shape[-1]
        train_r2_scores_per_feature = []
        for feat_idx in range(num_features):
            # Select masked data for the current feature
            actuals_feat = train_actuals_epoch[:, :, feat_idx][train_masks_epoch[:, :, feat_idx] > 0]
            preds_feat = train_predictions_epoch[:, :, feat_idx][train_masks_epoch[:, :, feat_idx] > 0]
            
            if len(actuals_feat) > 1 and np.var(actuals_feat) > 1e-6 : 
                train_r2_scores_per_feature.append(r2_score(actuals_feat, preds_feat))
            elif len(actuals_feat) > 1 and np.allclose(actuals_feat, preds_feat):
                 train_r2_scores_per_feature.append(1.0)
            else:
                train_r2_scores_per_feature.append(0.0) 
        
        avg_train_r2 = np.mean(train_r2_scores_per_feature) if train_r2_scores_per_feature else 0.0
        train_r2s.append(avg_train_r2)

        # --- Validation phase ---
        model.eval()
        batch_val_loss_sum = 0
        batch_val_mae_sum = 0
        total_val_elements = 0
        
        epoch_val_actuals_list = []
        epoch_val_predictions_list = []
        epoch_val_masks_list = [] # Initialize list for validation masks

        with torch.no_grad():
            for inputs, targets, input_masks_from_loader in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                input_masks_dev = input_masks_from_loader.to(device)
                
                outputs = model(inputs, input_mask=input_masks_dev)
                
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

        val_actuals_epoch = np.concatenate(epoch_val_actuals_list, axis=0)
        val_predictions_epoch = np.concatenate(epoch_val_predictions_list, axis=0)
        val_masks_epoch = np.concatenate(epoch_val_masks_list, axis=0)

        # R2 Calculation (Masked) for Validation Data
        # num_features is already defined from training actuals, assuming it's consistent
        val_r2_scores_per_feature = []
        for feat_idx in range(num_features):
            actuals_feat = val_actuals_epoch[:, :, feat_idx][val_masks_epoch[:, :, feat_idx] > 0]
            preds_feat = val_predictions_epoch[:, :, feat_idx][val_masks_epoch[:, :, feat_idx] > 0]

            if len(actuals_feat) > 1 and np.var(actuals_feat) > 1e-6:
                val_r2_scores_per_feature.append(r2_score(actuals_feat, preds_feat))
            elif len(actuals_feat) > 1 and np.allclose(actuals_feat, preds_feat):
                val_r2_scores_per_feature.append(1.0)
            else:
                val_r2_scores_per_feature.append(0.0)
        avg_val_r2 = np.mean(val_r2_scores_per_feature) if val_r2_scores_per_feature else 0.0
        # val_r2s.append(avg_val_r2) # Appending to val_r2s was duplicated. Corrected below.

        # The R2 calculation block you had at the end for validation was redundant
        # if the per-feature masked R2 is already calculated above.
        # The avg_val_r2 calculated from per-feature masked scores is the one to use.
        val_r2s.append(avg_val_r2) # Append the correct average R2 for validation

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train MAE: {avg_train_mae:.4f}, Val MAE: {avg_val_mae:.4f}, Train R2: {avg_train_r2:.4f}, Val R2: {avg_val_r2:.4f}')

    return train_losses, val_losses, train_maes, val_maes, train_r2s, val_r2s

# --- validate_model function also needs modification ---
def validate_model(model, data_loader, device):
    model.eval()
    predictions_list = [] # Use a list to collect batch outputs
    actuals_list = []     # Use a list to collect batch targets
    with torch.no_grad():
        for inputs, targets, input_masks in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs) # outputs shape: (batch, seq_len, features)
            predictions_list.append(outputs.cpu().numpy())
            actuals_list.append(targets.cpu().numpy()) # targets shape: (batch, seq_len, features)
            
    # Concatenate all batches along the 0-axis (batch dimension)
    predictions_array = np.concatenate(predictions_list, axis=0)
    actuals_array = np.concatenate(actuals_list, axis=0)
    
    return predictions_array, actuals_array

# --- analyze_results function (as discussed in the previous response) ---
# Make sure it handles 3D input arrays by reshaping or other appropriate logic.
# For example, reshaping to (total_timesteps, num_features)
def analyze_results(predictions, actuals, feature_names): # predictions/actuals are (num_samples, seq_len, num_features)
    num_samples, seq_len, num_features = actuals.shape
    
    predictions_flat = predictions.reshape(-1, num_features)
    actuals_flat = actuals.reshape(-1, num_features)

    mae = np.mean(np.abs(predictions_flat - actuals_flat), axis=0)
    mse = np.mean((predictions_flat - actuals_flat) ** 2, axis=0)
    rmse = np.sqrt(mse)

    r2_per_feature = []
    for i in range(num_features):
        ss_res = np.sum((actuals_flat[:, i] - predictions_flat[:, i]) ** 2)
        ss_tot = np.sum((actuals_flat[:, i] - np.mean(actuals_flat[:, i])) ** 2)
        
        if ss_tot == 0:
            # If total sum of squares is 0, R2 is 1 if residuals are also 0, otherwise it's problematic (can be 0 or undefined)
            # A common convention for r2_score for constant actuals is 1.0 if predictions are perfect, 0.0 otherwise.
            r2_val = 1.0 if ss_res == 0 else 0.0 
        else:
            r2_val = 1 - (ss_res / ss_tot)
        r2_per_feature.append(r2_val)

    results = pd.DataFrame({
        'Feature': feature_names,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2_per_feature
    })
    print("\nAnalysis Results (per feature, across all time steps):")
    print(results)
    return results

def plot_results(train_losses, val_losses, train_maes, val_maes, train_r2s, val_r2s):
    """Plots training and validation loss and MAE."""

    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # Plot Loss
    axs[0].plot(train_losses, label='Train Loss')
    axs[0].plot(val_losses, label='Validation Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Training and Validation Loss')
    axs[0].legend()

    # Plot MAE
    axs[1].plot(train_maes, label='Train MAE')
    axs[1].plot(val_maes, label='Validation MAE')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('MAE')
    axs[1].set_title('Training and Validation MAE')
    axs[1].legend()

    # Plot R²
    axs[2].plot(train_r2s, label='Train R²')
    axs[2].plot(val_r2s, label='Validation R²')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('R²')
    axs[2].set_title('Training and Validation R²')
    axs[2].legend()

    plt.tight_layout()
    plt.show()

