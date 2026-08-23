"""Train/val/test splitting with leakage prevention — frames from the same
source video/sequence must not end up split across train and test.
"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    # TODO: split by source video/sequence, not by individual frame, to
    # avoid near-duplicate frames leaking across the split boundary.
    raise NotImplementedError


if __name__ == "__main__":
    main()
