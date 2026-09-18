"""Champion-weight resolution for the inference service.

No model registry exists yet (Tier 1 scope) - the "champion" is whatever `.pt` file a GitHub
Release asset URL points at. `CARDD_MODEL_SOURCE=random-init` short-circuits to an
architecture-only, randomly-initialized model with no download and no network dependency at all -
this is what lets CI's Docker boot-smoke test run without a real trained weight ever existing in
CI (CarDD's license forbids any real weight/data artifact from being redistributed there).
"""
from __future__ import annotations

import os
from pathlib import Path

import requests

# The rolling "champion" release tag - its assets get replaced (not re-tagged) on each promotion
# (see .github/workflows/promote.yml), so this default never needs editing after this one change.
DEFAULT_CHAMPION_URL = "https://github.com/JShi12/vehicle-damage-detection/releases/download/champion/best.pt"
CACHE_PATH = Path(os.environ.get("CARDD_WEIGHTS_CACHE", "/tmp/cardd_champion.pt"))


def resolve_and_download_champion() -> str:
    """Returns a path/identifier suitable for `YOLO(...)`.

    - `CARDD_MODEL_SOURCE=random-init` -> "yolo11n.yaml" (no download, no trained weights: CI only)
    - `CARDD_MODEL_SOURCE=<local path>` -> that path, used as-is (local dev against a gitignored
      best.pt without needing any public hosting)
    - otherwise -> download `CHAMPION_WEIGHTS_URL` (or the baked-in default) once, cache it, and
      return the cached path
    """
    source = os.environ.get("CARDD_MODEL_SOURCE")
    if source == "random-init":
        return "yolo11n.yaml"
    if source:
        return source

    url = os.environ.get("CHAMPION_WEIGHTS_URL", DEFAULT_CHAMPION_URL)
    if not CACHE_PATH.exists():
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        CACHE_PATH.write_bytes(response.content)
    return str(CACHE_PATH)
