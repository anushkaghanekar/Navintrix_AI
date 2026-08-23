"""Prepare the emergency vehicle dataset (ambulance / fire_truck /
police_vehicle).

No single clean public dataset covers all three classes well — expect to
combine a few small Roboflow Universe sets plus manual curation. Document
in the report, per source: name, license, classes covered, image/video
count, annotation format. Budget this as its own multi-day task, not a
subtask of prepare_detrac.py.

Usage: python scripts/prepare_emergency.py --raw-dir data/raw/emergency --out-dir data/processed/emergency
"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    # TODO: merge sources, resolve class-name conflicts, convert everything
    # to a single consistent YOLO-format label set matching
    # configs/model.yaml's `classes.emergency`.
    raise NotImplementedError


if __name__ == "__main__":
    main()
