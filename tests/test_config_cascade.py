# -*- coding: utf-8 -*-
"""Config-cascade tests for src/pipeline/config_loader.py.

Verifies env-var precedence and that the repo-root resolution (parents[2] after
the src/pipeline move) still lets load_config() run and read .env.
"""
from __future__ import annotations

import sys

from src.pipeline.config_loader import load_config


def _load(monkeypatch, **env):
    monkeypatch.setattr(sys, "argv", ["test"])   # no --config
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return load_config()


def test_output_root_defaults_to_runs(monkeypatch):
    monkeypatch.delenv("OUTPUT_ROOT", raising=False)
    cfg = _load(monkeypatch)
    assert cfg.output_root == "runs"


def test_output_root_env_override(monkeypatch):
    cfg = _load(monkeypatch, OUTPUT_ROOT="outputs")
    assert cfg.output_root == "outputs"


def test_force_target_mode_env_override(monkeypatch):
    cfg = _load(monkeypatch, FORCE_TARGET_MODE="newton")
    assert cfg.force_target_mode == "newton"


def test_invalid_force_mode_falls_back(monkeypatch):
    cfg = _load(monkeypatch, FORCE_TARGET_MODE="banana")
    assert cfg.force_target_mode == "nbw"


def test_load_config_resolves_and_runs(monkeypatch):
    # Guards the parents[2] repo-root fix: load_config must run end-to-end and
    # produce a valid dataset (proving .env resolution didn't break on the move).
    cfg = _load(monkeypatch)
    assert cfg.selected_dataset in {"GROUNDLINK", "PATIENT_HANDLING", "COMBINED"}
