"""Generate the report's graphs/tables directly from experiment output
(evaluation/experiments.py results) — no numbers typed in by hand.
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def _extract_overall_avg_wait(metrics: dict) -> float:
    if "waiting_avg_by_road" in metrics and metrics["waiting_avg_by_road"]:
        waits = [w for w in metrics["waiting_avg_by_road"].values() if w > 0]
        return sum(waits) / len(waits) if waits else 0.0
    return float(metrics.get("average_waiting_time", 0.0))


def plot_waiting_time_comparison(results: dict, output_path: str) -> None:
    """Bar chart of average waiting time per controller per scenario,
    save to output_path.
    """
    scenarios: list[str] = []
    controllers = ["fixed_time", "density_only", "proposed"]
    data: dict[str, dict[str, float]] = {c: {} for c in controllers}

    for key, metrics in results.items():
        if isinstance(key, tuple) and len(key) == 2:
            ctrl, scn = key
            if scn not in scenarios:
                scenarios.append(scn)
            if ctrl in data:
                data[ctrl][scn] = _extract_overall_avg_wait(metrics)

    if not scenarios:
        scenarios = ["default"]

    x = np.arange(len(scenarios))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    offsets = [-width, 0, width]
    colors = ["#7f7f7f", "#1f77b4", "#2ca02c"]
    labels = ["Fixed-Time", "Density-Only", "Proposed (Adaptive)"]

    for i, ctrl in enumerate(controllers):
        values = [data[ctrl].get(s, 0.0) for s in scenarios]
        ax.bar(x + offsets[i], values, width, label=labels[i], color=colors[i])

    ax.set_xlabel("Scenario")
    ax.set_ylabel("Average Waiting Time (seconds)")
    ax.set_title("Controller Comparison: Average Waiting Time")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def plot_emergency_response(results: dict, output_path: str) -> None:
    """Chart of emergency response/clearance time, save to output_path."""
    scenarios: list[str] = []
    modes = ["emergency_priority_off", "emergency_priority_on"]
    data: dict[str, dict[str, float]] = {m: {} for m in modes}

    for key, metrics in results.items():
        if isinstance(key, tuple) and len(key) == 2:
            mode, scn = key
            if scn not in scenarios:
                scenarios.append(scn)
            val = float(metrics.get("emergency_active_steps", 0.0))
            if mode in data:
                data[mode][scn] = val

    if not scenarios:
        scenarios = ["default"]

    x = np.arange(len(scenarios))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    offsets = [-width / 2, width / 2]
    colors = ["#d62728", "#2ca02c"]
    labels = ["Priority OFF", "Priority ON"]

    for i, mode in enumerate(modes):
        values = [data[mode].get(s, 0.0) for s in scenarios]
        ax.bar(x + offsets[i], values, width, label=labels[i], color=colors[i])

    ax.set_xlabel("Scenario")
    ax.set_ylabel("Emergency Active / Clearance Steps")
    ax.set_title("Emergency Vehicle Response by Scenario")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
