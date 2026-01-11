from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROFILE_DIR = Path("config/profiles")


def load_profile(profile: str) -> dict[str, Any]:
    profile_path = PROFILE_DIR / f"{profile}.yaml"
    if not profile_path.exists():
        raise FileNotFoundError(f"Profile not found: {profile}")
    with profile_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_custom_profile() -> dict[str, Any]:
    custom_path = PROFILE_DIR / "custom.yaml"
    if not custom_path.exists():
        return {}
    with custom_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_custom_profile(config: dict[str, Any]) -> None:
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    custom_path = PROFILE_DIR / "custom.yaml"
    with custom_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
