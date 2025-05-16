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
    model.to(device)
    train_losses, val_losses = [], []
    train_maes, val_maes = [], []
    train_r2s, val_r2s = [], []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        batch_train_loss = 0
        batch_train_mae = 0
        
        epoch_train_actuals_list = []
        epoch_train_predictions_list = []

        for inputs, targets in train_loader: # targets shape: (batch_size, seq_len, num_features)
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)  # outputs shape: (batch_size, seq_len, num_features)
            
            # --- MODIFIED LOSS CALCULATION ---
            loss = criterion(outputs, targets) 
            loss.backward()
            optimizer.step()
            
            batch_train_loss += loss.item()

            # --- MODIFIED MAE CALCULATION ---
            mae = torch.mean(torch.abs(outputs - targets)).detach().cpu().numpy()
            batch_train_mae += mae

            # Store full sequences for epoch-level R2 calculation
            epoch_train_actuals_list.append(targets.cpu().numpy())
            epoch_train_predictions_list.append(outputs.detach().cpu().numpy())
            
        avg_train_loss = batch_train_loss / len(train_loader)
        avg_train_mae = batch_train_mae / len(train_loader)
        train_losses.append(avg_train_loss)
        train_maes.append(avg_train_mae)

        # Concatenate all batch results for the epoch
        train_actuals_epoch = np.concatenate(epoch_train_actuals_list, axis=0)
        train_predictions_epoch = np.concatenate(epoch_train_predictions_list, axis=0)

        # --- MODIFIED R2 CALCULATION (flattening time steps) ---
        num_features = train_actuals_epoch.shape[-1]
        train_actuals_flat = train_actuals_epoch.reshape(-1, num_features)
        train_predictions_flat = train_predictions_epoch.reshape(-1, num_features)
        
        # Handle potential constant actuals for R2 score stability
        if np.all(np.var(train_actuals_flat, axis=0) < 1e-6): # Check if all features have near-zero variance
             # If actuals are constant, R2 is 1 if predictions are also constant and equal, 0 otherwise by some conventions for this func
            avg_train_r2 = r2_score(train_actuals_flat, train_predictions_flat, multioutput='uniform_average') \
                           if np.allclose(train_actuals_flat, train_predictions_flat) else 0.0
        else:
            avg_train_r2 = r2_score(train_actuals_flat, train_predictions_flat, multioutput='uniform_average')
        train_r2s.append(avg_train_r2)

        # Validation phase (similar modifications needed)
        model.eval()
        batch_val_loss = 0
        batch_val_mae = 0
        epoch_val_actuals_list = []
        epoch_val_predictions_list = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                
                # --- MODIFIED LOSS CALCULATION ---
                loss = criterion(outputs, targets)
                batch_val_loss += loss.item()
                
                # --- MODIFIED MAE CALCULATION ---
                mae = torch.mean(torch.abs(outputs - targets)).detach().cpu().numpy()
                batch_val_mae += mae
                
                epoch_val_actuals_list.append(targets.cpu().numpy())
                epoch_val_predictions_list.append(outputs.detach().cpu().numpy())        

        avg_val_loss = batch_val_loss / len(val_loader)
        avg_val_mae = batch_val_mae / len(val_loader)
        val_losses.append(avg_val_loss)
        val_maes.append(avg_val_mae)

        val_actuals_epoch = np.concatenate(epoch_val_actuals_list, axis=0)
        val_predictions_epoch = np.concatenate(epoch_val_predictions_list, axis=0)

        # --- MODIFIED R2 CALCULATION (flattening time steps) ---
        val_actuals_flat = val_actuals_epoch.reshape(-1, num_features)
        val_predictions_flat = val_predictions_epoch.reshape(-1, num_features)

        if np.all(np.var(val_actuals_flat, axis=0) < 1e-6):
            avg_val_r2 = r2_score(val_actuals_flat, val_predictions_flat, multioutput='uniform_average') \
                         if np.allclose(val_actuals_flat, val_predictions_flat) else 0.0
        else:
            avg_val_r2 = r2_score(val_actuals_flat, val_predictions_flat, multioutput='uniform_average')
        val_r2s.append(avg_val_r2)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train MAE: {avg_train_mae:.4f}, Val MAE: {avg_val_mae:.4f}, Train R2: {avg_train_r2:.4f}, Val R2: {avg_val_r2:.4f}')

    return train_losses, val_losses, train_maes, val_maes, train_r2s, val_r2s

# --- validate_model function also needs modification ---
def validate_model(model, data_loader, device):
    model.eval()
    predictions_list = [] # Use a list to collect batch outputs
    actuals_list = []     # Use a list to collect batch targets
    with torch.no_grad():
        for inputs, targets in data_loader:
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

