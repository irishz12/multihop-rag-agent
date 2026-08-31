"""YAML config loading.

Config paths are resolved relative to the project root unless already
absolute, so scripts can be invoked from any working directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path = "configs/dataset.yaml") -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r") as f:
        return yaml.safe_load(f)
