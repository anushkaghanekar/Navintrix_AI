"""Load and validate configs/signal.yaml — single source of truth for
signal timing and controller coefficients so nothing gets hardcoded
elsewhere in controller/ or simulation/.
"""

import yaml


def load_signal_config(path: str = "configs/signal.yaml") -> dict:
    """TODO: load YAML, validate required keys are present
    (signal.min_green_seconds etc.), raise a clear error if not."""
    with open(path) as f:
        return yaml.safe_load(f)
