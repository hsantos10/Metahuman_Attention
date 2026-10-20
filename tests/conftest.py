# -*- coding: utf-8 -*-
"""Shared pytest configuration.

Ensures the repo root is importable so the absolute ``src.*`` package imports
resolve when tests run from anywhere.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
