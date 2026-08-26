"""Run the core experiments end-to-end and write raw results JSON.

Drives evaluation.experiments over the three committed SUMO scenarios and
writes the experiment_1_results.json / experiment_4_results.json that
evaluation/plots.py and the report consume. Requires a SUMO installation
and a built network (run scripts/build_sumo_scenarios.sh first).

  python scripts/run_experiments.py \
      --scenarios balanced heavy emergency \
      --max-steps 3600 \
      --out evaluation/results

If SUMO / TraCI is missing, the run fails cleanly with a message instead of
crashing — no numbers are fabricated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evaluation.experiments import run_experiment_1, run_experiment_4

DEFAULT_SCENARIOS = ["balanced", "heavy", "emergency"]
DEFAULT_CONFIG = "configs/signal.yaml"
DEFAULT_INTERSECTION_CONFIG = "configs/intersection.yaml"
DEFAULT_OUTPUT = "evaluation/results"


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenarios", nargs="*", default=DEFAULT_SCENARIOS,
                    help="scenario names; default balanced heavy emergency")
    ap.add_argument("--max-steps", type=int, default=2000,
                    help="simulation steps per run (1-second steps)")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--intersection-config", default=DEFAULT_INTERSECTION_CONFIG)
    ap.add_argument("--output", default=DEFAULT_OUTPUT,
                    help="directory for *_results.json")
    return ap


def main() -> int:
    args = _build_parser().parse_args()
    out = Path(args.output)

    try:
        results1 = run_experiment_1(
            args.scenarios,
            max_steps=args.max_steps,
            config_path=args.config,
            intersection_config_path=args.intersection_config,
            output_dir=str(out),
        )
        results4 = run_experiment_4(
            args.scenarios,
            max_steps=args.max_steps,
            config_path=args.config,
            intersection_config_path=args.intersection_config,
            output_dir=str(out),
        )
    except RuntimeError as exc:  # TraCI/SUMO missing or scenario misbuilt
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    def _brief(results, key):
        return {
            f"{c}::{(None if s is None else s)}": (
                dict(results[(c, s)]).get(key)
            )
            for c, s in sorted(results)
        }

    print("\n=== Experiment 1 (throughput_total, sim_seconds) ===")
    print(json.dumps({str(k): v for k, v in _brief(results1, "throughput_total").items()}, indent=2, sort_keys=True))
    print("\n=== Experiment 4 (emergency_active_steps) ===")
    print(json.dumps({str(k): v for k, v in _brief(results4, "emergency_active_steps").items()}, indent=2, sort_keys=True))
    print(f"\nRaw results written to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())