"""Per-channel z-score of the force/CoP targets, fitted on the training split's valid (masked) frames only:
`fit_target_normalizer` -> {"mean", "std"}, `apply_target_normalizer` on a split tuple, and the inverse."""
import numpy as np
import torch


def fit_target_normalizer(train_target_tensors, train_mask_tensors, target_columns):
    if not target_columns:
        return None

    target_indices = list(range(len(target_columns)))
    values_per_feature = [[] for _ in target_indices]

    for target_tensor, mask_tensor in zip(train_target_tensors, train_mask_tensors):
        valid_len = int(mask_tensor.sum().item())
        if valid_len <= 0:
            continue

        valid_targets = target_tensor[:valid_len, target_indices]
        for feature_idx in range(valid_targets.shape[1]):
            values_per_feature[feature_idx].append(valid_targets[:, feature_idx].detach().cpu().numpy())

    means = []
    stds = []
    for feature_chunks in values_per_feature:
        if not feature_chunks:
            means.append(0.0)
            stds.append(1.0)
            continue

        concatenated = np.concatenate(feature_chunks)
        mean_val = float(np.mean(concatenated))
        std_val = float(np.std(concatenated))
        std_val = std_val if std_val >= 1e-6 else 1.0
        means.append(mean_val)
        stds.append(std_val)

    return {
        'indices': target_indices,
        'mean': means,
        'std': stds,
        'target_columns': list(target_columns),
    }


def apply_target_normalizer(split_tuple, use_metadata, normalizer):
    if not split_tuple or not split_tuple[0] or normalizer is None:
        return split_tuple

    target_idx = 2 if use_metadata else 1
    mask_idx = 3 if use_metadata else 2

    trc_sequences = split_tuple[0]
    metadata_sequences = split_tuple[1] if use_metadata else None
    target_sequences = split_tuple[target_idx]
    mask_sequences = split_tuple[mask_idx]

    normalized_targets = []
    index_tensor = torch.tensor(normalizer['indices'], dtype=torch.long)
    mean_tensor = torch.tensor(normalizer['mean'], dtype=torch.float32)
    std_tensor = torch.tensor(normalizer['std'], dtype=torch.float32)

    for target_tensor, mask_tensor in zip(target_sequences, mask_sequences):
        normalized_target = target_tensor.clone()
        valid_len = int(mask_tensor.sum().item())
        if valid_len > 0:
            current_slice = normalized_target[:valid_len]
            current_slice[:, index_tensor] = (current_slice[:, index_tensor] - mean_tensor) / std_tensor
            normalized_target[:valid_len] = current_slice
        normalized_targets.append(normalized_target)

    if use_metadata:
        return (trc_sequences, metadata_sequences, normalized_targets, mask_sequences)
    return (trc_sequences, normalized_targets, mask_sequences)


def inverse_target_normalization(array_3d, masks_2d, normalizer):
    if normalizer is None:
        return array_3d

    restored = np.array(array_3d, copy=True)
    index_array = np.asarray(normalizer['indices'], dtype=np.int64)
    mean_array = np.asarray(normalizer['mean'], dtype=np.float32)
    std_array = np.asarray(normalizer['std'], dtype=np.float32)

    for sample_idx in range(restored.shape[0]):
        valid_len = int(np.sum(masks_2d[sample_idx])) if masks_2d is not None else restored.shape[1]
        if valid_len <= 0:
            continue

        sample_slice = restored[sample_idx, :valid_len, :]
        sample_slice[:, index_array] = (
            sample_slice[:, index_array] * std_array.reshape(1, -1) + mean_array.reshape(1, -1)
        )
        restored[sample_idx, :valid_len, :] = sample_slice

    return restored
