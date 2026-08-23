"""Prepare UA-DETRAC data — primary detector training/eval set for the 4
normal vehicle classes (car/bus/van-> map to your class set/truck/other).

Download (official host): https://detrac-db.rit.albany.edu/
Grab it early and keep a local copy — the official host is occasionally
slow. Roboflow Universe hosts several re-uploaded mirrors if needed as
backup, but verify their license/completeness against the original before
relying on one.

Usage: python scripts/prepare_detrac.py --raw-dir data/raw/detrac --out-dir data/processed/detrac
"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    # TODO: parse the XML annotations, map DETRAC's class set to
    # configs/model.yaml's `classes.normal`, convert to YOLO txt format.
    raise NotImplementedError


if __name__ == "__main__":
    main()
