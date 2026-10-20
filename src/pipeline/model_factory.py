# -*- coding: utf-8 -*-
"""
Model construction helpers.

Centralises ``build_model_from_params`` and ``torch.compile`` wrapping
so both the standard and LOSO pipelines share a single source of truth.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import (
    CNN_LSTM_Network,
    CNN_Network,
    ImprovedConvFormer,
    LSTM_Network_with_Attention,
)


# ── torch.compile wrapper ──────────────────────────────────────────────────

def maybe_compile(model: nn.Module) -> nn.Module:
    """Apply ``torch.compile`` when available (PyTorch >= 2.4, Linux/CUDA).

    Returns the model unchanged on unsupported platforms.
    """
    if not hasattr(torch, "compile"):
        return model
    if not torch.cuda.is_available():
        print("[torch.compile] Skipped — no CUDA device.")
        return model
    if sys.platform == "win32":
        print("[torch.compile] Skipped — Windows not supported.")
        return model
    try:
        compiled = torch.compile(model, mode="reduce-overhead")
        print("[torch.compile] Model compiled with mode='reduce-overhead'.")
        return compiled
    except Exception as exc:
        print(f"[torch.compile] Failed ({exc}), using eager mode.")
        return model


# ── Model factory ──────────────────────────────────────────────────────────


class LookaheadShift(nn.Module):
    """Gives a causal model a fixed look-ahead of L frames: the input (and its mask) is shifted L frames earlier,
    so the causal output at frame t is computed from input frames <= t + L. The targets and the loss mask are
    not shifted. L = 0 is plain causal. Frames past the end of the sequence are zero (as padding)."""

    def __init__(self, model: nn.Module, lookahead: int):
        super().__init__()
        self.model, self.lookahead = model, int(lookahead)

    def forward(self, x, metadata=None, input_mask=None):
        L = self.lookahead
        if L > 0:
            x = F.pad(x[:, L:], (0, 0, 0, L))
            if input_mask is not None:
                input_mask = F.pad(input_mask[:, L:], (0, L))
        return self.model(x, metadata, input_mask=input_mask)


def lookahead_from_env():
    """LOOKAHEAD env: unset/'full' -> None (non-causal, the default); an integer >= 0 -> that many frames."""
    v = os.getenv("LOOKAHEAD", "full").strip().lower()
    return None if v in ("", "full") else int(v)


def build_model(
    model_name: str,
    params: dict,
    input_size: int,
    output_size: int,
    max_length: int,
    device: torch.device,
) -> nn.Module:
    """Instantiate a model from *params* dict (as produced by Optuna).

    params["lookahead"] (int frames, or None = non-causal, the default) builds the model causal and wraps it in
    LookaheadShift, so the prediction at frame t uses input frames <= t + lookahead only."""

    lookahead = params.get("lookahead")
    causal = lookahead is not None

    if model_name == "LSTM_ATTENTION":
        if causal:
            raise ValueError("LSTM_ATTENTION has no causal variant")
        model = LSTM_Network_with_Attention(
            input_size=input_size,
            hidden_size=params["hidden_size"],
            output_size=output_size,
            num_layers=params["num_layers"],
            dropout=params["dropout"],
        )
    elif model_name == "CNN":
        model = CNN_Network(
            input_size, output_size,
            params["num_filters"], params["kernel_size"], params["dropout"], causal=causal,
        )
    elif model_name == "CNN_LSTM":
        model = CNN_LSTM_Network(
            input_size, params["hidden_size"], output_size,
            params["num_layers"], params["num_filters"],
            params["kernel_size"], params["dropout"], causal=causal,
        )
    elif model_name == "ImprovedConvFormer":
        model = ImprovedConvFormer(
            input_size=input_size,
            output_size=output_size,
            d_model=params["d_model"],
            nhead=params.get("nhead", 8),
            num_encoder_layers=params.get("num_encoder_layers", 2),
            dim_feedforward=params["dim_feedforward"],
            dropout=params["dropout"],
            num_cnn_filters=params["num_cnn_filters"],
            max_sequence_length=max_length,
            causal=causal,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    if causal:
        model = LookaheadShift(model, lookahead)
    return model.to(device)


def build_and_compile(
    model_name: str,
    params: dict,
    input_size: int,
    output_size: int,
    max_length: int,
    device: torch.device,
) -> nn.Module:
    """Build a model and optionally ``torch.compile`` it."""
    return maybe_compile(
        build_model(model_name, params, input_size, output_size, max_length, device)
    )
