from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROFILE_DIR = Path("config/profiles")


def parse_env_file(path: Path) -> dict[str, str]:
    """Parse a .env file and return key-value pairs."""
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.strip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def load_profile(profile: str) -> dict[str, Any]:
    profile_path = PROFILE_DIR / f"{profile}.yaml"
    if not profile_path.exists():
        raise FileNotFoundError(f"Profile not found: {profile}")
    with profile_path.open("r", encoding="utf-8") as handle:
        content = yaml.safe_load(handle)
        if content is None:
            return {}
        if not isinstance(content, dict):
            raise ValueError(f"Profile {profile} must be a YAML dictionary, got {type(content).__name__}")
        return content


def load_custom_profile() -> dict[str, Any]:
    custom_path = PROFILE_DIR / "custom.yaml"
    if not custom_path.exists():
        return {}
    with custom_path.open("r", encoding="utf-8") as handle:
        content = yaml.safe_load(handle)
        if content is None:
            return {}
        if not isinstance(content, dict):
            raise ValueError(f"Custom profile must be a YAML dictionary, got {type(content).__name__}")
        return content


def write_custom_profile(config: dict[str, Any]) -> None:
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    custom_path = PROFILE_DIR / "custom.yaml"
    with custom_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
