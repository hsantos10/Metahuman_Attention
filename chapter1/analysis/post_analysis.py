"""Chapter 1 post-training analysis: per-feature R^2 and error plots, attention-map extraction and
visualisation for ImprovedConvFormer, NBW/Newton conversions for reporting. Called by the standard pipeline
and the Chapter 1 analysis scripts; not used by Chapter 2."""
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.lines import Line2D
from torch.utils.data import DataLoader, TensorDataset
import ast
from sklearn.cluster import KMeans
from scipy.stats import entropy, binomtest, pearsonr

from src.model import LSTM_Network, LSTM_Network_with_Attention, CNN_Network, CNN_LSTM_Network, ImprovedConvFormer
from src.train import validate_model
from src.utils.conversions import (
    force_feature_indices as _force_feature_indices,
    metadata_weight_to_bw_newtons as _metadata_weight_to_bw_newtons,
    convert_nbw_targets_to_newtons as _convert_nbw_targets_to_newtons,
)

try:
    from chapter1.analysis.visualization import (
        plot_attention_heatmap_plotly as _viz_attn_heatmap,
        plot_overlaid_heatmap_and_signal as _viz_overlaid,
        plot_multi_head_grid as _viz_multi_head,
        plot_attention_evolution as _viz_attn_evolution,
        plot_prediction_quality as _viz_pred_quality,
        plot_performance_comparison as _viz_perf_comparison,
    )
    _HAS_PLOTLY_VIZ = True
except ImportError:
    _HAS_PLOTLY_VIZ = False


def _prepare_plot_dir(plot_dir):
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir


def _apply_plot_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'figure.dpi': 120,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'font.size': 11,
    })


def _save_plot(plot_dir, filename, dpi=180):
    save_path = os.path.join(plot_dir, filename)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    return save_path


def _downsample_attention(attn_map, max_points=400):
    if attn_map.shape[0] <= max_points:
        return attn_map
    step = max(1, int(np.ceil(attn_map.shape[0] / max_points)))
    return attn_map[::step, ::step]


def _robust_limits(values, lower_q=0.01, upper_q=0.99, margin_ratio=0.05):
    if len(values) == 0:
        return None
    lo = np.quantile(values, lower_q)
    hi = np.quantile(values, upper_q)
    if np.isclose(lo, hi):
        lo -= 1.0
        hi += 1.0
    span = hi - lo
    return lo - margin_ratio * span, hi + margin_ratio * span

def load_analysis_data(filepath='outputs/data/analysis_data.pt'):
    """Loads the data saved from the main training script."""
    if not os.path.exists(filepath):
        print(f"Error: Analysis data file not found at {filepath}")
        print("Please run main.py first to generate the data.")
        return None
    return torch.load(filepath, weights_only=False)


def inverse_plate2_targets(predictions, actuals, masks, normalizer):
    if normalizer is None:
        return predictions, actuals

    indices = np.asarray(normalizer.get('indices', []), dtype=np.int64)
    means = np.asarray(normalizer.get('mean', []), dtype=np.float32)
    stds = np.asarray(normalizer.get('std', []), dtype=np.float32)

    if indices.size == 0 or indices.size != means.size or indices.size != stds.size:
        return predictions, actuals

    predictions_restored = np.array(predictions, copy=True)
    actuals_restored = np.array(actuals, copy=True)

    for sample_idx in range(predictions_restored.shape[0]):
        valid_len = int(np.sum(masks[sample_idx])) if masks is not None else predictions_restored.shape[1]
        if valid_len <= 0:
            continue
        pred_slice = predictions_restored[sample_idx, :valid_len, :]
        act_slice = actuals_restored[sample_idx, :valid_len, :]

        pred_slice[:, indices] = pred_slice[:, indices] * stds.reshape(1, -1) + means.reshape(1, -1)
        act_slice[:, indices] = act_slice[:, indices] * stds.reshape(1, -1) + means.reshape(1, -1)

        predictions_restored[sample_idx, :valid_len, :] = pred_slice
        actuals_restored[sample_idx, :valid_len, :] = act_slice

    return predictions_restored, actuals_restored

def calculate_sparsity(attention_weights, threshold=0.01):
    """Calculates the percentage of attention weights below a threshold."""
    # attention_weights: (num_layers, num_heads, seq_len, seq_len)
    total_weights = attention_weights.numel()
    sparse_weights = (attention_weights < threshold).sum().item()
    return (sparse_weights / total_weights) * 100

def calculate_entropy(attention_weights):
    """Calculates Shannon entropy for attention distributions."""
    # attention_weights: (num_layers, num_heads, seq_len, seq_len)

    # Detach and Normalize
    attention_weights = attention_weights.detach()
    row_sums = attention_weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    attention_weights = attention_weights / row_sums

    # Move to cpu numpy
    weights_np = attention_weights.cpu().numpy()

    # Compute entropy along the key dimension (last axis)
    # H(p) = - sum p_i log p_i
    entropies = entropy(weights_np, axis=-1, base=2)

    return np.nanmean(entropies)

def calculate_temporal_distance(attention_weights):
    """Calculates average temporal distance between query and attended keys."""
    # attention_weights: (num_layers, num_heads, seq_len, seq_len)
    num_layers, num_heads, seq_len, _ = attention_weights.shape
    device = attention_weights.device

    # Create index matrices
    q_indices = torch.arange(seq_len, device=device).unsqueeze(1) # (seq_len, 1)
    k_indices = torch.arange(seq_len, device=device).unsqueeze(0) # (1, seq_len)

    distances = torch.abs(q_indices - k_indices).float() # (seq_len, seq_len)

    # Weighted average distance
    # Expand distances to match attention weights shape
    distances = distances.unsqueeze(0).unsqueeze(0).expand(num_layers, num_heads, seq_len, seq_len)

    weighted_distances = (attention_weights * distances).sum(dim=-1) # Sum over keys -> (L, H, S)

    return weighted_distances.mean().item()

def detect_anchors(attention_weights, top_percent=0.05):
    """Identifies the most attended timesteps (keys)."""
    # Aggregate attention over queries, heads, layers to find most attended keys
    # attention_weights: (L, H, Q, K)
    # Sum over L, H, Q -> (K)
    key_importance = attention_weights.sum(dim=(0, 1, 2))

    # Identify top 5%
    k = max(1, int(len(key_importance) * top_percent))
    top_values, top_indices = torch.topk(key_importance, k)

    return top_indices.cpu().numpy(), key_importance.cpu().numpy()

def analyze_phases(attention_weights, force_signal, engage_threshold=10, release_threshold=10):
    """Segments trial and computes attention flow per phase."""
    # force_signal: (seq_len,)

    force_mag = np.abs(force_signal)
    max_force = np.max(force_mag)
    if max_force == 0: max_force = 1 # Avoid div by zero

    # Normalized thresholds
    norm_force = force_mag / max_force

    # Define phases indices
    active_indices = np.where(norm_force > 0.1)[0]

    if len(active_indices) == 0:
        return {}, {}, (None, None, None)

    start_idx = active_indices[0]
    end_idx = active_indices[-1]
    peak_idx = np.argmax(norm_force)

    # Engage: Start to Peak
    engage_mask = np.zeros_like(norm_force, dtype=bool)
    engage_mask[start_idx:peak_idx] = True

    # Release: Peak to End
    release_mask = np.zeros_like(norm_force, dtype=bool)
    release_mask[peak_idx:end_idx] = True

    # Hold: Let's define it as a window around peak
    hold_window = int(len(norm_force) * 0.1) # 10% window
    hold_start = max(start_idx, peak_idx - hold_window // 2)
    hold_end = min(end_idx, peak_idx + hold_window // 2)
    hold_mask = np.zeros_like(norm_force, dtype=bool)
    hold_mask[hold_start:hold_end] = True

    # Refine Engage/Release to exclude Hold
    engage_mask = engage_mask & (~hold_mask)
    release_mask = release_mask & (~hold_mask)

    phases = {'Engage': engage_mask, 'Hold': hold_mask, 'Release': release_mask}

    # Compute attention flow: Where do queries in each phase attend to?
    # attention_weights: (L, H, Q, K)
    # Aggregate to (Q, K) first
    avg_attn = attention_weights.mean(dim=(0, 1)).cpu().numpy() # (Q, K)

    phase_attention = {}
    for phase_name, mask in phases.items():
        if not np.any(mask):
            phase_attention[phase_name] = 0
            continue

        # Select rows (Queries) belonging to this phase
        # Ensure mask length matches query length
        if len(mask) > avg_attn.shape[0]:
             mask = mask[:avg_attn.shape[0]]

        phase_rows = avg_attn[mask, :]
        if phase_rows.size == 0:
            phase_attention[phase_name] = 0
            continue

        # Where do they attend? Average distribution over keys
        avg_dist = phase_rows.mean(axis=0)
        phase_attention[phase_name] = avg_dist

    return phases, phase_attention, (start_idx, peak_idx, end_idx)

def cluster_heads(attention_weights):
    """Clusters attention heads by behavioral metrics.

    Features per head:
    - Sparsity (% below threshold)
    - Entropy (distribution spread)
    - Temporal Distance (local vs global attention)
    """
    # attention_weights: (L, H, Q, K)
    num_layers, num_heads, _, _ = attention_weights.shape

    head_features = []
    head_labels = []

    for l in range(num_layers):
        for h in range(num_heads):
            attn = attention_weights[l, h] # (Q, K)

            # Feature 1: Sparsity
            sparsity = calculate_sparsity(attn)

            # Feature 2: Entropy
            head_entropy = calculate_entropy(attn.unsqueeze(0).unsqueeze(0))

            # Feature 3: Temporal Distance
            temp_dist = calculate_temporal_distance(attn.unsqueeze(0).unsqueeze(0))

            head_features.append([sparsity, head_entropy, temp_dist])
            head_labels.append(f"L{l+1}H{h+1}")

    head_features = np.array(head_features)

    # KMeans Clustering
    n_clusters = min(3, len(head_features))
    if n_clusters > 1:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(head_features)
    else:
        clusters = np.zeros(len(head_features))

    return head_features, clusters, head_labels


def infer_cluster_role(center_sparsity, center_entropy, center_temp_distance, global_temp_mean):
    """Assign interpretable role names from cluster center metrics."""
    if center_sparsity >= 96.0 and center_entropy <= 6.0:
        return "Feature Extractor"
    if center_sparsity <= 95.0 and center_entropy >= 6.3:
        return "Contextualizer"
    if center_temp_distance >= global_temp_mean:
        return "Global Integrator"
    return "Local Tracker"

def perform_core_analyses(model, data_loader, device, target_feature_names, model_name="ImprovedConvFormer", plot_dir='outputs/plots'):
    """Executes the suite of core analyses requested across the entire dataset."""
    print(f"\nPerforming Core Analyses for {model_name} (Population Analysis)...")

    model.eval()

    # Accumulators for population metrics
    all_sparsity = []
    all_entropy = []
    all_temp_dist = []

    # For printing mean stats
    epoch_entropies = []
    epoch_maes = []

    population_key_importance = None

    # Binomial Test Accumulators
    total_anchors = 0
    anchors_in_roi = 0
    total_roi_coverage = 0
    num_valid_samples = 0

    sample_for_viz = None
    sample_viz_masks = None
    sample_viz_targets = None

    gallery_data = [] # For multi-sample gallery

    vy_index = target_feature_names.index('ground_force_1_vert')

    for batch_idx, (inputs, metadata, targets, masks) in enumerate(data_loader):
        inputs = inputs.to(device)
        metadata = metadata.to(device)
        targets = targets.to(device)
        masks = masks.to(device)

        with torch.no_grad():
            outputs = model(inputs, metadata, input_mask=masks, return_attention=True)
            attention_weights = model.attention_weights
            if attention_weights.dim() == 4:
                 attention_weights = attention_weights.unsqueeze(2)

        if batch_idx == 0:
            sample_for_viz = attention_weights[:, 0, :, :, :].clone()
            sample_viz_masks = masks[0].clone()
            sample_viz_targets = targets[0].clone()

        batch_size = attention_weights.shape[1]

        # Calculate MAE per sample for reporting
        abs_diff = torch.abs(outputs - targets) # (B, S, F)
        mask_expanded = masks.unsqueeze(-1).expand_as(abs_diff)
        # sum over S and F, divide by valid count
        mae_per_sample = (abs_diff * mask_expanded).sum(dim=(1, 2)) / mask_expanded.sum(dim=(1, 2)).clamp(min=1e-9)

        for b in range(batch_size):
            valid_len = int(masks[b].sum().item())
            if valid_len == 0: continue

            attn = attention_weights[:, b, :, :valid_len, :valid_len]

            # Metrics
            all_sparsity.append(calculate_sparsity(attn))
            ent = calculate_entropy(attn)
            all_entropy.append(ent)
            all_temp_dist.append(calculate_temporal_distance(attn))

            # Correlation Data (Mean Reporting)
            epoch_entropies.append(ent)
            epoch_maes.append(mae_per_sample[b].item())

            # Anchor Detection
            sample_key_imp = attn.sum(dim=(0, 1, 2)).cpu()

            if population_key_importance is None:
                population_key_importance = torch.zeros(valid_len, device='cpu')

            if valid_len > len(population_key_importance):
                # Resize
                new_storage = torch.zeros(valid_len, device='cpu')
                new_storage[:len(population_key_importance)] = population_key_importance
                population_key_importance = new_storage

            population_key_importance[:valid_len] += sample_key_imp[:valid_len]

            # --- ROI Analysis for Binomial Test ---
            # Identify individual sample anchors (Top 5%)
            top_indices, _ = detect_anchors(attn, top_percent=0.05)

            # Define ROI: Peak Force +/- 20, Start +/- 20, End +/- 20
            force_signal = targets[b, :valid_len, vy_index].cpu().numpy()
            _, _, events = analyze_phases(attn, force_signal)
            start_idx, peak_idx, end_idx = events

            # Collect data for gallery (first 4 valid samples)
            if len(gallery_data) < 4:
                gallery_data.append({
                    'force': force_signal,
                    'anchors': top_indices,
                    'events': events,
                    'sample_id': f"Batch{batch_idx}_Sample{b}"
                })

            if all(x is not None for x in [start_idx, peak_idx, end_idx]):
                roi_mask = np.zeros(valid_len, dtype=bool)
                window = 20 # +/- 20 frames

                # Add Peak
                p_min, p_max = max(0, peak_idx - window), min(valid_len, peak_idx + window)
                roi_mask[p_min:p_max] = True
                # Add Start
                s_min, s_max = max(0, start_idx - window), min(valid_len, start_idx + window)
                roi_mask[s_min:s_max] = True
                # Add End
                e_min, e_max = max(0, end_idx - window), min(valid_len, end_idx + window)
                roi_mask[e_min:e_max] = True

                # Count Hits
                hits = np.sum(roi_mask[top_indices])

                total_anchors += len(top_indices)
                anchors_in_roi += hits
                total_roi_coverage += (np.sum(roi_mask) / valid_len)
                num_valid_samples += 1

    # --- 1. Quantitative Metrics ---
    print("\n1. Quantitative Metrics (Population Average)")
    print(f"Sparsity: {np.mean(all_sparsity):.2f}% ± {np.std(all_sparsity):.2f}% (Target: >95%)")
    print(f"Entropy: {np.mean(all_entropy):.2f} ± {np.std(all_entropy):.2f} bits (Target: ~2.3)")
    print(f"Temporal Distance: {np.mean(all_temp_dist):.2f} ± {np.std(all_temp_dist):.2f} steps")

    # --- 2. Anchor Detection ---
    print("\n2. Anchor Detection (Population Level)")
    if population_key_importance is not None:
        top_indices_pop, _ = torch.topk(population_key_importance, max(1, int(len(population_key_importance)*0.05)))
        print(f"Top 5% Most Attended Timesteps (Population Accumulation - Sampled): {top_indices_pop[:20].numpy()}...")

    # --- 3. Binomial Test for Structure ---
    print("\n3. Binomial Test for Structure")
    if num_valid_samples > 0:
        avg_expected_prob = total_roi_coverage / num_valid_samples
        # Test: Is observed probability (anchors_in_roi / total_anchors) significantly > avg_expected_prob?
        res = binomtest(anchors_in_roi, total_anchors, avg_expected_prob, alternative='greater')
        print(f"Anchors in ROI (Peak/Boundaries +/- 20): {anchors_in_roi}/{total_anchors} ({anchors_in_roi/total_anchors*100:.1f}%)")
        print(f"Expected Random Probability: {avg_expected_prob*100:.1f}%")
        print(f"Binomial Test p-value: {res.pvalue:.2e} (Target: <0.001)")
        if res.pvalue < 0.001:
            print("Result: Anchors are significantly structured around biomechanical events.")
        else:
            print("Result: Anchors are not significantly structured.")

    # --- 4. Phase-Dependent Dynamics (Representative Sample) ---
    print("\n4. Phase-Dependent Dynamics (Representative Sample)")
    valid_len_sample = int(sample_viz_masks.sum().item())
    force_signal_sample = sample_viz_targets[:valid_len_sample, vy_index].cpu().numpy()
    attn_sample = sample_for_viz[:, :, :valid_len_sample, :valid_len_sample]

    phases, phase_attn, _ = analyze_phases(attn_sample, force_signal_sample)

    print("Attention Flow per Phase (Top attended regions):")
    for phase, dist in phase_attn.items():
        if isinstance(dist, int): continue
        top_phase_k = np.argsort(dist)[-3:]
        print(f"  {phase}: Attends mostly to indices {top_phase_k}")

    # --- 5. Multi-Head Specialization (Representative Sample) ---
    print("\n5. Multi-Head Specialization (Representative Sample)")
    head_features, clusters, head_labels = cluster_heads(attn_sample)
    cluster_counts = pd.Series(clusters).value_counts().sort_index()
    redundancy_ratio = cluster_counts.max() / len(head_labels)

    for i in range(len(head_labels)):
        print(
            f"  Head {head_labels[i]}: Cluster {clusters[i]}, "
            f"Sparsity={head_features[i][0]:.1f}%, Entropy={head_features[i][1]:.2f}, "
            f"TemporalDistance={head_features[i][2]:.1f}"
        )

    print("\n  Cluster counts:")
    for cluster_id, count in cluster_counts.items():
        print(f"    Cluster {cluster_id}: {count} heads")
    print(f"  Redundancy ratio (largest cluster / all heads): {redundancy_ratio:.2f}")

    # --- 6. Attention vs Performance ---
    print("\n6. Attention vs Performance")
    print(f"Mean Attention Entropy: {np.mean(epoch_entropies):.2f} bits")
    print(f"Mean Prediction Error (MAE): {np.mean(epoch_maes):.4f}")
    print("Note: Entropy measures focus, not accuracy.")

    plot_dir = _prepare_plot_dir(plot_dir)

    _apply_plot_style()

    # --- 7. Visualizations ---
    print("\nGenerating Core Visualizations...")
    # 7.1 Aligned Heatmaps (Force + Attention)
    plot_overlaid_heatmap_and_signal(
        model,
        data_loader,
        device,
        target_feature_names,
        sample_idx=0,
        model_name=model_name,
        plot_dir=plot_dir,
    )

    # 7.2 Multi-Head Attention Comparison (Layer 1)
    plot_multi_head_grid(
        model,
        data_loader,
        device,
        sample_idx=0,
        model_name=model_name,
        layer_to_show=0,
        plot_dir=plot_dir,
    )

    # 7.3 Population Anchor Density
    if population_key_importance is not None:
        plt.figure(figsize=(12, 4.2))
        pop_vals = population_key_importance.numpy()
        plt.plot(pop_vals, color='tab:blue', linewidth=1.8)
        plt.fill_between(np.arange(len(pop_vals)), pop_vals, alpha=0.18, color='tab:blue')
        plt.title("Population Anchor Density (Key Importance)")
        plt.xlabel("Timestep")
        plt.ylabel("Accumulated Attention")
        plt.grid(alpha=0.3)
        anchor_density_path = _save_plot(plot_dir, f"anchor_density_{model_name}.png")
        print(f"Anchor density plot saved to {anchor_density_path}")

    # 7.4 Multi-Sample Anchor Gallery (Figure 9)
    if gallery_data:
        num_samples_gal = len(gallery_data)
        fig, axes = plt.subplots(num_samples_gal, 1, figsize=(12, 3 * num_samples_gal), sharex=False)
        if num_samples_gal == 1: axes = [axes]

        for i, data in enumerate(gallery_data):
            ax = axes[i]
            force = data['force']
            anchors = data['anchors']
            start, peak, end = data['events']

            ax.plot(force, label='Force Signal', color='tab:blue', linewidth=1.8)
            # Plot anchors as 'x' markers on the force signal
            ax.scatter(anchors, force[anchors], color='tab:red', marker='x', s=28, label='Anchors', zorder=5)

            if peak is not None:
                ax.axvline(peak, color='tab:orange', linestyle='--', alpha=0.6, label='Peak')
            if start is not None:
                ax.axvline(start, color='tab:green', linestyle=':', alpha=0.6, label='Start')
            if end is not None:
                ax.axvline(end, color='tab:purple', linestyle=':', alpha=0.6, label='End')

            ax.set_title(f"Sample {data['sample_id']}")
            ax.set_ylabel("Force")
            ax.grid(alpha=0.25)
            if i == 0: ax.legend()

        anchor_gallery_path = _save_plot(plot_dir, f"anchor_gallery_{model_name}.png")
        print(f"Anchor gallery saved to {anchor_gallery_path}")

    # 7.5 Head Specialization Schematic (Figure 10) - prioritized view
    unique_clusters = np.unique(clusters)
    # Fixed palette requested: blue, green, red
    base_cluster_colors = ['#1f77b4', '#2ca02c', '#d62728']
    cluster_color_map = {
        int(c): base_cluster_colors[i % len(base_cluster_colors)]
        for i, c in enumerate(unique_clusters)
    }

    fig, ax_scatter = plt.subplots(1, 1, figsize=(13.5, 9.5))

    # Scatter Plot (Sparsity vs Temporal Distance, entropy encoded by marker size)
    sparsity_vals = head_features[:, 0]
    entropy_vals = head_features[:, 1]
    temp_dist_vals = head_features[:, 2]

    if len(entropy_vals) > 1 and np.max(entropy_vals) > np.min(entropy_vals):
        entropy_norm = (entropy_vals - np.min(entropy_vals)) / (np.max(entropy_vals) - np.min(entropy_vals) + 1e-9)
        marker_sizes = 230 + 420 * entropy_norm
    else:
        marker_sizes = np.full_like(entropy_vals, 360.0)

    layer_ids = np.array([int(lbl.split('H')[0].replace('L', '')) for lbl in head_labels])
    unique_layers = sorted(np.unique(layer_ids))
    layer_markers = ['o', '*', '^', 'D', 'P', 'X']

    scatter = None
    for layer_idx, layer_id in enumerate(unique_layers):
        layer_mask = layer_ids == layer_id
        layer_marker = layer_markers[layer_idx % len(layer_markers)]
        layer_sizes = marker_sizes[layer_mask].copy()
        if layer_id == 2:
            layer_sizes = layer_sizes * 1.48

        marker_edgewidth = 0.55 if layer_id == 1 else 0.9
        point_colors = [cluster_color_map[int(c)] for c in clusters[layer_mask]]
        scatter = ax_scatter.scatter(
            sparsity_vals[layer_mask],
            temp_dist_vals[layer_mask],
            c=point_colors,
            s=layer_sizes,
            marker=layer_marker,
            edgecolor='black',
            linewidth=marker_edgewidth,
            alpha=0.92,
            label=f'Layer {layer_id}',
        )

    ax_scatter.set_facecolor('#f2f4f8')
    ax_scatter.set_xlabel('Sparsity (%)')
    ax_scatter.set_ylabel('Temporal Distance')
    ax_scatter.set_title('Multi-Head Specialization Clusters (Primary View)')
    ax_scatter.grid(alpha=0.72, color='#ffffff', linewidth=1.6)

    point_ids = np.arange(1, len(head_labels) + 1)
    for i, point_id in enumerate(point_ids):
        ax_scatter.annotate(
            str(point_id),
            (sparsity_vals[i], temp_dist_vals[i]),
            textcoords='offset points',
            xytext=(0, 0),
            fontsize=9.5,
            fontweight='bold',
            ha='center',
            va='center',
            color='black',
            bbox=dict(boxstyle='circle,pad=0.24', facecolor='white', edgecolor='none', alpha=0.9),
        )

    layer_legend_handles = [
        Line2D([0], [0], marker=layer_markers[idx % len(layer_markers)], color='w',
               markerfacecolor='lightgray', markeredgecolor='black', markeredgewidth=0.8,
               markersize=(13 if layer_id == 2 else 11), linestyle='None', label=f'Layer {layer_id}')
        for idx, layer_id in enumerate(unique_layers)
    ]

    # Cluster center role callouts
    cluster_summary_rows = []
    temp_mean = float(np.mean(temp_dist_vals))
    for cluster_id in unique_clusters:
        idx = np.where(clusters == cluster_id)[0]
        if len(idx) == 0:
            continue

        center_s = float(np.mean(sparsity_vals[idx]))
        center_e = float(np.mean(entropy_vals[idx]))
        center_t = float(np.mean(temp_dist_vals[idx]))
        role = infer_cluster_role(center_s, center_e, center_t, temp_mean)
        ax_scatter.scatter(center_s, center_t, marker='X', s=220, c='black', edgecolor='white', linewidth=0.8, zorder=6)
        ax_scatter.text(
            center_s,
            center_t,
            f" C{int(cluster_id)}",
            fontsize=10,
            fontweight='bold',
            va='bottom',
            ha='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='none'),
        )

        cluster_summary_rows.append({
            'cluster_id': int(cluster_id),
            'role': role,
            'num_heads': int(len(idx)),
            'mean_sparsity': center_s,
            'mean_entropy': center_e,
            'mean_temporal_distance': center_t,
        })

    # Cluster role legend (C0/C1/...) for interpretation
    cluster_legend_handles = []
    for row in cluster_summary_rows:
        cluster_id = row['cluster_id']
        role = row['role']
        num_heads = row['num_heads']
        cluster_color = cluster_color_map.get(int(cluster_id), (0.5, 0.5, 0.5))

        cluster_legend_handles.append(
            Line2D(
                [0], [0], marker='o', color='w', linestyle='None',
                markerfacecolor=cluster_color, markeredgecolor='black', markeredgewidth=0.8,
                markersize=11, label=f"C{cluster_id}: {role} (n={num_heads})"
            )
        )

    if cluster_legend_handles:
        cluster_legend = ax_scatter.legend(
            handles=cluster_legend_handles,
            loc='upper left',
            bbox_to_anchor=(0.01, 1.0),
            frameon=True,
            title='Cluster Role',
            fontsize=8.8,
        )
        ax_scatter.add_artist(cluster_legend)

    layer_legend = ax_scatter.legend(
        handles=layer_legend_handles,
        loc='upper left',
        bbox_to_anchor=(0.01, 0.76),
        frameon=True,
        title='Layer',
        fontsize=9,
    )
    ax_scatter.add_artist(layer_legend)

    head_spec_path = _save_plot(plot_dir, f"head_specialization_schematic_{model_name}.png")
    print(f"Head specialization schematic saved to {head_spec_path}")

    # Save quantitative specialization summary for reporting
    summary_df = pd.DataFrame(cluster_summary_rows).sort_values('cluster_id')
    summary_csv = os.path.join(plot_dir, f"head_specialization_summary_{model_name}.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"Head specialization summary saved to {summary_csv}")

    print("Core Analysis Complete.")


def plot_attention_heatmap(model, data_loader, device, sample_idx=0, model_name="", plot_dir='outputs/plots', max_heatmap_points=400):
    """Plots the attention heatmap for a single sample from the test set."""
    model.eval()

    inputs, metadata, targets, masks = next(iter(data_loader))
    input_sample = inputs[sample_idx].unsqueeze(0).to(device)
    meta_sample = metadata[sample_idx].unsqueeze(0).to(device)
    mask_sample = masks[sample_idx].unsqueeze(0).to(device)

    with torch.no_grad():
        if hasattr(model, 'forward'):
             _ = model(input_sample, meta_sample, input_mask=mask_sample, return_attention=True)

        if not hasattr(model, 'attention_weights') or model.attention_weights is None:
            print(f"Model {model_name} does not support attention visualization or weights are None.")
            return

        attention_weights = model.attention_weights

        # Handle batch dimension in attention_weights if present
        if attention_weights.dim() == 5:
             attention_weights = attention_weights[:, 0]
        elif attention_weights.dim() == 4:
             attention_weights = attention_weights[:, 0].unsqueeze(1)

    num_layers, num_heads, seq_len, _ = attention_weights.shape

    valid_len = int(mask_sample.sum().item())
    attention_weights = attention_weights[:, :, :valid_len, :valid_len]
    attn_np = attention_weights.cpu().numpy()

    # --- Plotly fast-path ---
    if _HAS_PLOTLY_VIZ:
        try:
            _viz_attn_heatmap(
                attn_np, num_layers, num_heads,
                model_name=model_name, sample_idx=sample_idx,
                save_dir=plot_dir, max_heatmap_points=max_heatmap_points,
                downsample_fn=_downsample_attention,
            )
            print(f"Attention heatmap (plotly) saved to {plot_dir}")
            return
        except Exception as exc:
            print(f"Plotly attention heatmap failed ({exc}), falling back to matplotlib.")

    # --- Matplotlib fallback ---
    _apply_plot_style()
    fig, axes = plt.subplots(num_layers, num_heads, figsize=(4.1 * num_heads, 4.1 * num_layers), sharex=True, sharey=True)
    fig.suptitle(f'Attention Heatmaps for {model_name} - Sample {sample_idx}', fontsize=24)

    if num_layers == 1 and num_heads == 1:
        axes = np.array([[axes]])
    elif num_layers == 1:
        axes = np.expand_dims(axes, axis=0)
    elif num_heads == 1:
        axes = np.expand_dims(axes, axis=1)

    for i in range(num_layers):
        for j in range(num_heads):
            ax = axes[i, j]
            attn_2d = attn_np[i, j]
            attn_2d = _downsample_attention(attn_2d, max_points=max_heatmap_points)
            vmax = np.quantile(attn_2d, 0.995) if attn_2d.size > 0 else None
            sns.heatmap(attn_2d, ax=ax, cmap='viridis', cbar=False, vmin=0.0, vmax=vmax)
            ax.set_title(f'Layer {i+1}, Head {j+1}')
            ax.set_xticks([])
            ax.set_yticks([])

    plot_dir = _prepare_plot_dir(plot_dir)
    attn_heatmap_path = _save_plot(plot_dir, f'attention_heatmap_{model_name}_sample_{sample_idx}.png')
    print(f"Attention heatmap saved to {attn_heatmap_path}")


def plot_overlaid_heatmap_and_signal(model, data_loader, device, target_feature_names, sample_idx=0, model_name="", layer_to_show=1, head_to_show=0, plot_dir='outputs/plots', max_heatmap_points=450):
    """Creates a composite visualization with GRF signal and attention heatmap."""
    model.eval()

    inputs, metadata, targets, masks = next(iter(data_loader))
    input_sample = inputs[sample_idx].unsqueeze(0).to(device)
    meta_sample = metadata[sample_idx].unsqueeze(0).to(device)
    mask_sample = masks[sample_idx].unsqueeze(0).to(device)

    with torch.no_grad():
        if hasattr(model, 'forward'):
            _ = model(input_sample, meta_sample, input_mask=mask_sample, return_attention=True)

        if not hasattr(model, 'attention_weights') or model.attention_weights is None:
            print("Cannot generate overlaid plot: attention weights not available.")
            return

        attention_weights = model.attention_weights
        if attention_weights.dim() == 5:
            attention_weights = attention_weights[:, 0]
        elif attention_weights.dim() == 4:
            attention_weights = attention_weights[:, 0].unsqueeze(1)

    valid_len = int(mask_sample.sum().item())

    num_layers, num_heads, _, _ = attention_weights.shape
    if layer_to_show >= num_layers: layer_to_show = 0
    if head_to_show >= num_heads: head_to_show = 0

    # Get the specific attention map
    attn_map = attention_weights[layer_to_show, head_to_show, :valid_len, :valid_len].cpu().numpy()
    attn_map = _downsample_attention(attn_map, max_points=max_heatmap_points)

    # Get the vertical GRF signal
    vy_index = target_feature_names.index('ground_force_1_vert')
    vy_signal = targets[sample_idx, :valid_len, vy_index].cpu().numpy()

    # --- Plotly fast-path ---
    if _HAS_PLOTLY_VIZ:
        try:
            _viz_overlaid(
                attn_map, vy_signal, model_name,
                sample_idx=sample_idx, layer_idx=layer_to_show, head_idx=head_to_show,
                save_dir=plot_dir,
            )
            print(f"Overlaid attention plot (plotly) saved to {plot_dir}")
            return
        except Exception as exc:
            print(f"Plotly overlaid plot failed ({exc}), falling back to matplotlib.")

    # --- Matplotlib fallback ---
    _apply_plot_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10.2), gridspec_kw={'height_ratios': [1, 2]})

    # Top Panel: Vertical Force Signal
    ax1.plot(vy_signal, color='tab:blue', linewidth=1.8)
    ax1.set_title(f'Vertical Ground Reaction Force (vert) - Sample {sample_idx}')
    ax1.set_ylabel('Force (N)')
    ax1.axvline(x=0, color='r', linestyle='--', alpha=0.6, label='Start')
    ax1.axvline(x=valid_len-1, color='g', linestyle='--', alpha=0.6, label='End')
    ax1.legend()
    ax1.grid(True)

    # Bottom Panel: Attention Heatmap
    vmax = np.quantile(attn_map, 0.995) if attn_map.size > 0 else None
    sns.heatmap(attn_map, ax=ax2, cmap='viridis', cbar=True, vmin=0.0, vmax=vmax)
    ax2.set_title(f'Attention Heatmap (Layer {layer_to_show+1}, Head {head_to_show+1})')
    ax2.set_xlabel('Key Sequence Timesteps')
    ax2.set_ylabel('Query Sequence Timesteps')
    ax2.text(0.02, 0.98, f"Downsampled to {attn_map.shape[0]}x{attn_map.shape[1]}", transform=ax2.transAxes,
             va='top', ha='left', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='none'))

    plot_dir = _prepare_plot_dir(plot_dir)
    overlaid_path = _save_plot(plot_dir, f'overlaid_attention_{model_name}_sample_{sample_idx}.png')
    print(f"Overlaid attention plot saved to {overlaid_path}")


def plot_multi_head_grid(model, data_loader, device, sample_idx=0, model_name="", layer_to_show=1, plot_dir='outputs/plots', max_heatmap_points=350):
    """Creates a grid showing all attention heads from a specific layer."""
    model.eval()

    inputs, metadata, targets, masks = next(iter(data_loader))
    input_sample = inputs[sample_idx].unsqueeze(0).to(device)
    meta_sample = metadata[sample_idx].unsqueeze(0).to(device)
    mask_sample = masks[sample_idx].unsqueeze(0).to(device)

    with torch.no_grad():
        if hasattr(model, 'forward'):
            _ = model(input_sample, meta_sample, input_mask=mask_sample, return_attention=True)

        if not hasattr(model, 'attention_weights') or model.attention_weights is None:
            print("Cannot generate multi-head grid: attention weights not available.")
            return

        attention_weights = model.attention_weights
        if attention_weights.dim() == 5:
            attention_weights = attention_weights[:, 0]
        elif attention_weights.dim() == 4:
            attention_weights = attention_weights[:, 0].unsqueeze(1)

    num_layers, num_heads, _, _ = attention_weights.shape
    valid_len = int(mask_sample.sum().item())

    if layer_to_show >= num_layers:
        print(f"Layer {layer_to_show+1} is out of bounds. Model has {num_layers} layers. Using Layer 1.")
        layer_to_show = 0

    layer_attns = attention_weights[layer_to_show, :, :valid_len, :valid_len].cpu().numpy()

    # --- Plotly fast-path ---
    if _HAS_PLOTLY_VIZ:
        try:
            _viz_multi_head(
                layer_attns, num_heads, layer_to_show,
                model_name=model_name, sample_idx=sample_idx,
                save_dir=plot_dir, max_heatmap_points=max_heatmap_points,
                downsample_fn=_downsample_attention,
            )
            print(f"Multi-head grid (plotly) saved to {plot_dir}")
            return
        except Exception as exc:
            print(f"Plotly multi-head grid failed ({exc}), falling back to matplotlib.")

    # --- Matplotlib fallback ---
    # Dynamic grid size based on num_heads
    cols = 4
    rows = (num_heads + cols - 1) // cols
    _apply_plot_style()
    fig, axes = plt.subplots(rows, cols, figsize=(4.1*cols, 4.1*rows), sharex=True, sharey=True)
    fig.suptitle(f'Multi-Head Attention Comparison - Layer {layer_to_show+1}', fontsize=24)

    # Flatten axes for easy iteration
    axes_flat = axes.flat if num_heads > 1 else [axes]

    for i in range(len(axes_flat)):
        ax = axes_flat[i]
        if i < num_heads:
            head_map = _downsample_attention(layer_attns[i], max_points=max_heatmap_points)
            vmax = np.quantile(head_map, 0.995) if head_map.size > 0 else None
            sns.heatmap(head_map, ax=ax, cmap='viridis', cbar=False, vmin=0.0, vmax=vmax)
            ax.set_title(f'Head {i+1}')
        else:
            ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])

    plot_dir = _prepare_plot_dir(plot_dir)
    multi_head_path = _save_plot(plot_dir, f'multi_head_grid_{model_name}_layer_{layer_to_show+1}_sample_{sample_idx}.png')
    print(f"Multi-head grid plot saved to {multi_head_path}")


def plot_attention_evolution(model, data_loader, device, sample_idx=0, model_name="", head_to_show=0, plot_dir='outputs/plots', max_heatmap_points=400):
    """Shows the evolution of an attention head across layers."""
    model.eval()

    inputs, metadata, targets, masks = next(iter(data_loader))
    input_sample = inputs[sample_idx].unsqueeze(0).to(device)
    meta_sample = metadata[sample_idx].unsqueeze(0).to(device)
    mask_sample = masks[sample_idx].unsqueeze(0).to(device)

    with torch.no_grad():
        if hasattr(model, 'forward'):
            _ = model(input_sample, meta_sample, input_mask=mask_sample, return_attention=True)

        if not hasattr(model, 'attention_weights') or model.attention_weights is None:
            print("Cannot generate attention evolution plot: attention weights not available.")
            return

        attention_weights = model.attention_weights
        if attention_weights.dim() == 5:
            attention_weights = attention_weights[:, 0]
        elif attention_weights.dim() == 4:
             attention_weights = attention_weights[:, 0].unsqueeze(1)

    num_layers, num_heads, _, _ = attention_weights.shape
    valid_len = int(mask_sample.sum().item())

    if head_to_show >= num_heads:
        print(f"Head {head_to_show+1} is out of bounds. Model has {num_heads} heads.")
        return

    attn_np = attention_weights.cpu().numpy()

    # --- Plotly fast-path ---
    if _HAS_PLOTLY_VIZ:
        try:
            _viz_attn_evolution(
                attn_np, num_layers, head_to_show, valid_len,
                model_name=model_name, sample_idx=sample_idx,
                save_dir=plot_dir, max_heatmap_points=max_heatmap_points,
                downsample_fn=_downsample_attention,
            )
            print(f"Attention evolution (plotly) saved to {plot_dir}")
            return
        except Exception as exc:
            print(f"Plotly attention evolution failed ({exc}), falling back to matplotlib.")

    # --- Matplotlib fallback ---
    _apply_plot_style()
    fig, axes = plt.subplots(1, num_layers, figsize=(6.8 * num_layers, 5.8), sharey=True)
    if num_layers == 1:
        axes = [axes]
    fig.suptitle(f'Attention Evolution for Head {head_to_show+1}', fontsize=20)

    for i in range(num_layers):
        attn_map = attn_np[i, head_to_show, :valid_len, :valid_len]
        attn_map = _downsample_attention(attn_map, max_points=max_heatmap_points)
        vmax = np.quantile(attn_map, 0.995) if attn_map.size > 0 else None
        sns.heatmap(attn_map, ax=axes[i], cmap='viridis', cbar=i==num_layers-1, vmin=0.0, vmax=vmax)
        axes[i].set_title(f'Layer {i+1}')
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    plot_dir = _prepare_plot_dir(plot_dir)
    attn_evo_path = _save_plot(plot_dir, f'attention_evolution_{model_name}_head_{head_to_show+1}_sample_{sample_idx}.png')
    print(f"Attention evolution plot saved for {model_name}, head {head_to_show+1}: {attn_evo_path}")


def plot_prediction_quality(predictions, actuals, target_feature_names, model_name, sample_idx=0, masks=None, plot_dir='outputs/plots'):
    """Plots predictions vs. actuals for a single sample."""

    # --- Plotly fast-path ---
    if _HAS_PLOTLY_VIZ:
        try:
            _viz_pred_quality(
                predictions, actuals, target_feature_names, model_name,
                sample_idx=sample_idx, masks=masks, save_dir=plot_dir,
            )
            print(f"Prediction quality plot (plotly) saved to {plot_dir}")
            return
        except Exception as exc:
            print(f"Plotly prediction quality failed ({exc}), falling back to matplotlib.")

    # --- Matplotlib fallback ---
    num_features = len(target_feature_names)
    fig, axes = plt.subplots(num_features, 1, figsize=(15, 4 * num_features), sharex=True)
    if num_features == 1:
        axes = [axes] # Make it iterable

    if masks is not None:
        valid_len = int(np.sum(masks[sample_idx]))
    else:
        valid_len = actuals.shape[1]

    _apply_plot_style()

    for i in range(num_features):
        y_true = actuals[sample_idx, :valid_len, i]
        y_pred = predictions[sample_idx, :valid_len, i]
        axes[i].plot(y_true, label='Actual', color='tab:blue', linewidth=1.7)
        axes[i].plot(y_pred, label='Predicted', color='tab:red', linestyle='--', linewidth=1.5)
        lims = _robust_limits(np.concatenate([y_true, y_pred]), lower_q=0.005, upper_q=0.995, margin_ratio=0.08)
        if lims is not None:
            axes[i].set_ylim(*lims)
        axes[i].set_ylabel(target_feature_names[i])
        axes[i].legend()
        axes[i].set_title(f'{model_name} - {target_feature_names[i]}')
        axes[i].grid(alpha=0.25)

    axes[-1].set_xlabel('Timestep')
    plt.suptitle(f'Prediction vs. Actual for Sample {sample_idx}')
    plot_dir = _prepare_plot_dir(plot_dir)
    pred_quality_path = _save_plot(plot_dir, f'prediction_quality_{model_name}_sample_{sample_idx}.png')
    print(f"Prediction quality plot saved to {pred_quality_path}")


def plot_performance_comparison(results_df, plot_dir='outputs/plots'):
    """Plots bar charts comparing model performance metrics."""

    # --- Plotly fast-path ---
    if _HAS_PLOTLY_VIZ:
        try:
            _viz_perf_comparison(results_df, save_dir=plot_dir)
            print(f"Performance comparison (plotly) saved to {plot_dir}")
            return
        except Exception as exc:
            print(f"Plotly performance comparison failed ({exc}), falling back to matplotlib.")

    # --- Matplotlib fallback ---
    _apply_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    sns.barplot(x='model', y='test_mae', data=results_df, ax=axes[0], color='tab:blue')
    axes[0].set_title('Test MAE Comparison')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(alpha=0.25, axis='y')

    sns.barplot(x='model', y='test_r2', data=results_df, ax=axes[1], color='tab:green')
    axes[1].set_title('Test R-squared Comparison')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].axhline(0.75, color='tab:red', linestyle='--', linewidth=1.2, alpha=0.8, label='Target R²=0.75')
    axes[1].legend(loc='best')
    axes[1].grid(alpha=0.25, axis='y')

    plot_dir = _prepare_plot_dir(plot_dir)
    performance_path = _save_plot(plot_dir, 'performance_comparison.png')
    print(f"Performance comparison plot saved to {performance_path}")


def main(run_root='outputs', results_file=None, model_dir=None, analysis_data_path=None, generate_auxiliary_heatmaps=False):
    """Main function to run the post-analysis."""
    torch.manual_seed(42)
    np.random.seed(42)
    print("--- Starting Post-Analysis ---")

    run_root = os.path.normpath(run_root)
    if analysis_data_path is None:
        analysis_data_path = os.path.join(run_root, 'data', 'analysis_data.pt')
    if results_file is None:
        results_file = os.path.join(run_root, 'results', 'model_comparison_results.csv')
    if model_dir is None:
        model_dir = os.path.join(run_root, 'models')
    plot_dir = os.path.join(run_root, 'plots', 'post_analysis')

    analysis_data = load_analysis_data(analysis_data_path)
    if analysis_data is None:
        return

    test_dataset = analysis_data['test_dataset']
    target_feature_names = analysis_data['target_feature_names']
    models_to_run = analysis_data['models_to_run']
    input_size = analysis_data['input_size']
    output_size = analysis_data['output_size']
    max_length = analysis_data.get('max_length', 5000)
    plate2_target_normalizer = analysis_data.get('target_normalizer', analysis_data.get('plate2_target_normalizer'))
    force_target_mode = str(analysis_data.get('force_target_mode', 'newton')).lower()

    metadata_np = None
    if hasattr(test_dataset, 'tensors') and len(test_dataset.tensors) > 1:
        try:
            metadata_np = test_dataset.tensors[1].cpu().numpy()
        except Exception:
            metadata_np = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # batch_size=4 to match training, avoiding shape mismatch in batch-dependent processing if any
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    if os.path.exists(results_file):
        results_df = pd.read_csv(results_file)
        plot_performance_comparison(results_df, plot_dir=plot_dir)
    else:
        print(f"Warning: Results file '{results_file}' not found. Skipping performance comparison plot.")
        results_df = None

    for model_name in models_to_run:
        print(f"\n--- Analyzing model: {model_name} ---")
        model_path = os.path.join(model_dir, f"{model_name}_model.pt")
        if not os.path.exists(model_path):
            print(f"Warning: Model file not found at {model_path}. Skipping analysis for this model.")
            continue

        if results_df is None:
            print(f"Cannot instantiate model {model_name} because results file was not found. Skipping.")
            continue

        model_hyperparams_row = results_df[results_df['model'] == model_name]
        if model_hyperparams_row.empty:
            print(f"Cannot find hyperparameters for {model_name} in {results_file}. Skipping.")
            continue

        try:
            hyperparams = ast.literal_eval(model_hyperparams_row['best_hyperparameters'].iloc[0])
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing hyperparameters for {model_name}: {e}. Skipping.")
            continue

        model = None
        if model_name == "ImprovedConvFormer":
             model = ImprovedConvFormer(
                input_size=input_size,
                output_size=output_size,
                d_model=hyperparams['d_model'],
                nhead=hyperparams.get('nhead', 8),
                num_encoder_layers=hyperparams.get('num_encoder_layers', 2),
                dim_feedforward=hyperparams['dim_feedforward'],
                dropout=hyperparams['dropout'],
                num_cnn_filters=hyperparams['num_cnn_filters'],
                max_sequence_length=max_length
            )
        else:
            print(f"Skipping analysis for non-transformer model: {model_name}")
            continue

        loaded_model_data = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(loaded_model_data, dict) and 'model_state_dict' in loaded_model_data:
            model.load_state_dict(loaded_model_data['model_state_dict'])
        else:
            model.load_state_dict(loaded_model_data)
        model.to(device)

        predictions, actuals, masks = validate_model(model, test_loader, device)
        predictions, actuals = inverse_plate2_targets(predictions, actuals, masks, plate2_target_normalizer)

        if force_target_mode == 'nbw':
            force_indices = _force_feature_indices(target_feature_names)
            predictions = _convert_nbw_targets_to_newtons(predictions, metadata_np, force_indices)
            actuals = _convert_nbw_targets_to_newtons(actuals, metadata_np, force_indices)

        if predictions.shape[0] == 0:
            print(f"No predictions were generated for {model_name}. Skipping plots.")
            continue

        plot_prediction_quality(
            predictions,
            actuals,
            target_feature_names,
            model_name,
            sample_idx=0,
            masks=masks,
            plot_dir=plot_dir,
        )

        if model_name == "ImprovedConvFormer":
            # Call the new core analysis function
            perform_core_analyses(
                model,
                test_loader,
                device,
                target_feature_names,
                model_name=model_name,
                plot_dir=plot_dir,
            )

            # Optional auxiliary raw heatmaps (de-emphasized by default)
            if generate_auxiliary_heatmaps:
                plot_multi_head_grid(model, test_loader, device, sample_idx=0, model_name=model_name, plot_dir=plot_dir)
                plot_attention_evolution(model, test_loader, device, sample_idx=0, model_name=model_name, plot_dir=plot_dir)

    print("\n--- Post-Analysis Complete ---")


if __name__ == '__main__':
    main()
