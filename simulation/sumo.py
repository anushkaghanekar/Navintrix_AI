"""SUMO network/config helpers.

The actual .net.xml / .rou.xml / .sumocfg files for the four-way
intersection and each scenario live in simulation/scenarios/ — build them
with SUMO's netedit/netconvert and duflow/randomTrips tools, then reference
them here. This module is the Python-side helper for loading/validating
those files, not a from-scratch SUMO network generator.
"""


def load_scenario_config(scenario_name: str) -> str:
    """TODO: return the path to simulation/scenarios/{scenario_name}.sumocfg,
    raising a clear error if it doesn't exist yet.
    """
    raise NotImplementedError
