"""Prepare AI City Challenge 2021 Track 1 data.

Used for counting/movement validation, NOT as primary detector training
data — it's a camera-viewpoint/ROI/movement-annotation dataset, treat it
as such rather than as an ordinary image classification set.

Download: https://www.aicitychallenge.org/2021-track1-download/
(no data-request form needed for this specific dataset as of the org's
current dataset-access page — re-check before assuming that's still true).

Usage: python scripts/prepare_aicity.py --raw-dir data/raw/aicity --out-dir data/processed/aicity
"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    # TODO: preserve video/camera/ROI/movement-annotation relationships —
    # do not flatten into a plain image folder.
    raise NotImplementedError


if __name__ == "__main__":
    main()
