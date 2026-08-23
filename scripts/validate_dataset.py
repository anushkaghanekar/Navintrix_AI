"""Sanity-check a prepared dataset before training: corrupt images,
missing labels, out-of-range boxes, class distribution, and a random
sample of boxes drawn on images for visual verification.
"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--sample-visual-checks", type=int, default=20)
    args = parser.parse_args()
    # TODO: check every image has a matching label file (or is
    # legitimately empty), every box is within image bounds, print class
    # distribution, save `sample_visual_checks` annotated images to
    # data/processed/_visual_check/ for manual review.
    raise NotImplementedError


if __name__ == "__main__":
    main()
