"""Generate the report's graphs/tables directly from experiment output
(evaluation/experiments.py results) — no numbers typed in by hand.
"""

import matplotlib.pyplot as plt


def plot_waiting_time_comparison(results: dict, output_path: str) -> None:
    """TODO: bar chart of average waiting time per controller per scenario,
    save to output_path.
    """
    raise NotImplementedError


def plot_emergency_response(results: dict, output_path: str) -> None:
    """TODO: chart of emergency response/clearance time, save to output_path."""
    raise NotImplementedError
