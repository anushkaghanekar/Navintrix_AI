"""Deterministic route generator for the three core SUMO scenarios.

Produces the per-scenario .rou.xml files under simulation/scenarios/routes/
from fixed parameters (no randomTrips call, so runs are reproducible across
machines — ROADMAP requires fixed seeds). The FSM/run loop reads vehicles
from TraCI; these files only say WHO shows up WHEN on the four approaches.

Four approach routes (single-lane straight crossings, right-hand traffic):
    N->S : [N0, S1]   S->N : [S0, N1]   E->W : [E0, W1]   W->E : [W0, E1]

Scenarios:
    balanced   moderate, roughly even demand on all four approaches
    heavy      high demand on all approaches (queue build-up + length)
    emergency  light normal traffic + scheduled ambulances that conflict with
               the FSM's currently-granted axis, so emergency priority fires.

Usage:
    python scripts/generate_sumo_routes.py [--out simulation/scenarios/routes]
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_OUTDIR = "simulation/scenarios/routes"

ROUTES = {
    "N->S": ("N0", "S1"),
    "S->N": ("S0", "N1"),
    "E->W": ("E0", "W1"),
    "W->E": ("W0", "E1"),
}

VTYPES = {
    "car": 'id="car" vClass="passenger"',
    "truck": 'id="truck" vClass="truck"',
    "bus": 'id="bus" vClass="bus"',
    "ambulance": 'id="ambulance" vClass="emergency"',
}

# per-approach spawn counts and headway per scenario
DEMAND = {
    "balanced": (30, 20.0),
    "heavy": (150, 4.0),
    "emergency": (20, 30.0),
}


def _vtype_block(with_ambulance: bool) -> str:
    names = ["car", "truck", "bus"] + (["ambulance"] if with_ambulance else [])
    return "".join(f"    <vType {VTYPES[n]}/>\n" for n in names)


def _vehicle(vid: str, vtype: str, depart: float, edges: str) -> str:
    return (
        f'    <vehicle id="{vid}" type="{vtype}" depart="{depart:.1f}">'
        f'<route edges="{edges}"/></vehicle>\n'
    )


def build(scenario: str) -> str:
    body = _vtype_block(scenario == "emergency")
    count, headway = DEMAND[scenario]

    for name in ROUTES:
        e0, e1 = ROUTES[name]
        for i in range(count):
            vtype = "car"
            if scenario == "heavy":
                vtype = "truck" if i % 6 == 0 else ("bus" if i % 11 == 0 else "car")
            depart = 10.0 + i * headway
            body += _vehicle(f"v_{name}_{i}", vtype, depart, f"{e0} {e1}")

    if scenario == "emergency":
        # One ambulance per approach, staggered so several arrive while a
        # *different* axis holds green -> emergency priority conflicts.
        for i, name in enumerate(ROUTES):
            e0, e1 = ROUTES[name]
            depart = 120.0 + i * 900.0
            body += _vehicle(f"em_{name}_0", "ambulance", depart, f"{e0} {e1}")
    return body


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUTDIR)
    args = ap.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    for scenario in ("balanced", "heavy", "emergency"):
        text = f"<routes>\n{build(scenario)}</routes>\n"
        (outdir / f"{scenario}.rou.xml").write_text(text, encoding="utf-8")
    print(f"wrote balanced/heavy/emergency .rou.xml under {outdir}/")


if __name__ == "__main__":
    main()