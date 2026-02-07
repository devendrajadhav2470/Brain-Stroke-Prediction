"""YAML configuration loader."""

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Parameters
    ----------
    config_path : str or Path
        Path to the YAML config file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def load_all_configs(config_dir: str | Path = "configs") -> dict[str, Any]:
    """Load all YAML configs from the config directory and merge them.

    Parameters
    ----------
    config_dir : str or Path
        Directory containing YAML config files.

    Returns
    -------
    dict
        Merged configuration dictionary.
    """
    config_dir = Path(config_dir)
    merged = {}

    for config_file in sorted(config_dir.glob("*.yaml")):
        config = load_config(config_file)
        merged.update(config)

    return merged

