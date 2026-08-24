"""Load and validate configs/signal.yaml — single source of truth for
signal timing and controller coefficients so nothing gets hardcoded
elsewhere in controller/ or simulation/.
"""

from __future__ import annotations

import yaml

DEFAULT_CONFIG_PATH = "configs/signal.yaml"

_REQUIRED_SIGNAL = (
    "min_green_seconds",
    "max_green_seconds",
    "yellow_seconds",
    "all_red_seconds",
)


def load_signal_config(path: str = DEFAULT_CONFIG_PATH) -> dict:
    """Load YAML and validate the required keys are present
    (signal.min_green_seconds etc.), raising a clear error if not.
    Returns the full parsed config dict; callers use cfg['signal']."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or "signal" not in cfg:
        raise ValueError(f"{path}: missing 'signal' section")
    signal = cfg["signal"]
    if not isinstance(signal, dict):
        raise ValueError(f"{path}: 'signal' must be a mapping")
    missing = [k for k in _REQUIRED_SIGNAL if k not in signal]
    if missing:
        raise ValueError(f"{path}: signal section missing required key(s): {', '.join(missing)}")
    return cfg
