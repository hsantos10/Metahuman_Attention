# -*- coding: utf-8 -*-
"""
Custom loss functions for force prediction models.

This module implements specialized loss functions that address data challenges:
- Coordinate mismatch between motion capture and force plate systems
- Scale imbalance between Force (Newtons) and CoP (meters) data
- Model tendency to flatten predictions for subtle signals
"""

import torch
import torch.nn as nn


class NormalizedPeakWeightedLoss(nn.Module):
    """
    A weighted MSE loss that emphasizes peaks in the target data while balancing 
    scales across features with vastly different ranges.
    
    Why Normalized Peak Weighting?
    -----
    Standard loss functions (MSE, Huber) treat all errors equally. When predicting
    both Forces (typical range 0-400 N) and Center of Pressure (typical range 0-0.15 m),
    the raw scales create a 2600x imbalance. This causes:
    - Model focuses primarily on Force predictions
    - CoP predictions become flat/smoothed (model plays it safe)
    
    Solution:
    - Normalize each feature within the batch to [0, 1] range
    - Weight errors by normalized target magnitude (peaks matter more)
    - Result: A "peak" in Force and a "peak" in CoP receive equal emphasis
    
    Mathematics:
    -----
    For each feature f in each sample:
        min_f = minimum value in batch for feature f
        max_f = maximum value in batch for feature f
        range_f = max_f - min_f + epsilon
        target_normalized = (target - min_f) / (max_f - min_f + epsilon)
        error_normalized = (prediction - target) / range_f
        weight = 1 + multiplier * target_normalized
        loss = weight * (error_normalized)^2
    
    The multiplier (default 2.0) controls peak emphasis:
    - 0.0: No peak emphasis (uniform loss)
    - 2.0: Peaks weighted 3x more than flat regions
    - 5.0: Peaks weighted 6x more than flat regions
    """
    
    def __init__(self, peak_weight_multiplier=2.0, epsilon=1e-9, feature_weights=None, min_range=1e-6):
        """
        Initialize the normalized peak-weighted loss.
        
        Args:
            peak_weight_multiplier (float): How much more to emphasize peaks.
                - 0.0 = no emphasis (standard MSE)
                - 2.0 = peaks get 3x weight (recommended)
                - 5.0 = peaks get 6x weight (very aggressive)
            epsilon (float): Small value to prevent division by zero.
            feature_weights (list[float] | None): Optional static weight per output
                feature (length = num_features). Applied after normalized peak
                weighting to rebalance feature importance.
        """
        super().__init__()
        self.peak_weight_multiplier = peak_weight_multiplier
        self.epsilon = epsilon
        self.feature_weights = feature_weights
        self.min_range = min_range
    
    def forward(self, predictions, targets):
        """
        Compute normalized peak-weighted loss (element-wise or scalar).
        
        Args:
            predictions (torch.Tensor): Model predictions, shape (batch, seq_len, num_features)
            targets (torch.Tensor): Ground truth targets, shape (batch, seq_len, num_features)
        
        Returns:
            torch.Tensor: Element-wise loss with shape (batch, seq_len, num_features)
                         allowing downstream masking and per-feature weighting
        """
        # Ensure inputs have the same shape
        if predictions.shape != targets.shape:
            raise ValueError(
                f"Predictions shape {predictions.shape} does not match "
                f"targets shape {targets.shape}"
            )
        
        # Normalize each feature within the batch
        # Reshape for feature-wise operations: (batch * seq_len, num_features)
        batch_size, seq_len, num_features = targets.shape
        targets_flat = targets.reshape(-1, num_features)
        predictions_flat = predictions.reshape(-1, num_features)
        
        # Compute min/max per feature across the batch
        targets_min = targets_flat.min(dim=0, keepdim=True)[0]
        targets_max = targets_flat.max(dim=0, keepdim=True)[0]
        
        # Normalize to [0, 1] range per feature
        raw_range = targets_max - targets_min
        safe_range = torch.where(raw_range.abs() < self.min_range,
                     torch.ones_like(raw_range),
                     raw_range + self.epsilon)

        targets_normalized = (targets_flat - targets_min) / safe_range

        # Normalize residuals by feature range (fixes force-vs-CoP scale imbalance)
        normalized_error = (predictions_flat - targets_flat) / safe_range
        normalized_squared_error = normalized_error ** 2
        
        # Compute weights: 1 + multiplier * normalized_magnitude
        # At minimum: weight = 1.0 (flat regions)
        # At maximum: weight = 1 + multiplier (peaks, emphasis applied)
        weights = 1.0 + self.peak_weight_multiplier * targets_normalized
        
        # Apply weights to normalized squared error (element-wise)
        weighted_squared_error = normalized_squared_error * weights

        if self.feature_weights is not None:
            if len(self.feature_weights) != num_features:
                raise ValueError(
                    f"feature_weights length {len(self.feature_weights)} does not match "
                    f"num_features {num_features}"
                )
            feature_weight_tensor = torch.tensor(
                self.feature_weights,
                dtype=weighted_squared_error.dtype,
                device=weighted_squared_error.device,
            ).unsqueeze(0)
            weighted_squared_error = weighted_squared_error * feature_weight_tensor
        
        # Reshape back to (batch, seq_len, num_features) for masking compatibility
        return weighted_squared_error.reshape(batch_size, seq_len, num_features)


class MultiScaleWeightedLoss(nn.Module):
    """
    Alternative loss that applies explicit per-feature weights for highly imbalanced tasks.
    
    Use this if you want direct control over which features get emphasized.
    Example: Force features get weight 1.0, CoP features get weight 2.0
    """
    
    def __init__(self, feature_weights=None):
        """
        Initialize multi-scale weighted loss.
        
        Args:
            feature_weights (list): Weights for each output feature.
                Example: [1.0, 1.0, 5.0, 1.0, 1.0, 1.0, 1.0, 5.0, 1.0, 1.0]
                for 10 features where indices 2 and 7 (Z-axis) get 5x weight
        """
        super().__init__()
        self.feature_weights = feature_weights
    
    def forward(self, predictions, targets):
        """
        Compute multi-scale weighted MSE loss.
        
        Args:
            predictions (torch.Tensor): Model predictions, shape (batch, seq_len, num_features)
            targets (torch.Tensor): Ground truth targets, shape (batch, seq_len, num_features)
        
        Returns:
            torch.Tensor: Scalar loss value
        """
        squared_error = (predictions - targets) ** 2
        
        if self.feature_weights is not None:
            weights_tensor = torch.tensor(
                self.feature_weights,
                dtype=squared_error.dtype,
                device=squared_error.device
            ).unsqueeze(0).unsqueeze(0)
            squared_error = squared_error * weights_tensor
        
        return squared_error.mean()
