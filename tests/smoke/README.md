# tests/smoke/

Hand-run check on synthetic data. It executes at module level, so pytest ignores this folder
(`--ignore=tests/smoke` in `pyproject.toml`).

```bash
python tests/smoke/test_bilateral_flip.py     # left-right mirror augmentation: markers, rel features, forces, CoP
```

Unit tests live one level up: `python -m pytest tests`.
