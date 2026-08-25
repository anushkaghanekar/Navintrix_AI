"""SUMO network/config helpers.

The actual .net.xml / .rou.xml / .sumocfg files for the four-way
intersection and each scenario live in simulation/scenarios/ — build them
with SUMO's netedit/netconvert and duflow/randomTrips tools, then reference
them here. This module is the Python-side helper for loading/validating
those files, not a from-scratch SUMO network generator.
"""

from __future__ import annotations

from pathlib import Path

# simulation/scenarios/ relative to this file (simulation/sumo.py -> scenarios/).
SCENARIO_DIR = Path(__file__).resolve().parent / "scenarios"


def load_scenario_config(scenario_name: str) -> str:
    """Return the path to simulation/scenarios/{scenario_name}.sumocfg,
    raising a clear error if it doesn't exist yet.

    Scenarios are built with netedit/netconvert + duflow/randomTrips (see the
    module docstring); this helper only locates and validates the resulting
    config so the rest of the code can rely on its existence.
    """
    if not scenario_name or "/" in scenario_name or "\\" in scenario_name or scenario_name in (".", ".."):
        raise ValueError(f"invalid scenario name: {scenario_name!r}")
    path = SCENARIO_DIR / f"{scenario_name}.sumocfg"
    if not path.is_file():
        raise FileNotFoundError(
            f"scenario config not found: {path}. Build the scenario with "
            "netedit/netconvert and place its .sumocfg under simulation/scenarios/."
        )
    return str(path)


def available_scenarios() -> list[str]:
    """Names of scenarios that currently have a .sumocfg under scenarios/."""
    return sorted(p.stem for p in SCENARIO_DIR.glob("*.sumocfg"))
