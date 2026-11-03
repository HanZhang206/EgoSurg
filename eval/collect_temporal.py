#!/usr/bin/env python3
# Copyright 2025 The EgoSurg Authors.
# SPDX-License-Identifier: Apache-2.0
"""Aggregate temporal_evaluation/summary_mean.csv into Table 3B.

Each CSV is one 10-frame clip. The paper table is the mean and sample standard
deviation (ddof=1) over surgeries 1-6 and phases 01-05 (30 clips). A subset
that drops surgeries 5 and 6 does not reproduce the published t-PSNR.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from statistics import mean, stdev

METHODS = (
    ("pointmap", "Depth reprojection"),
    ("3DGS_random", "Image-only 3DGS"),
    ("3DGS_rgpd", "EgoSurg w/o diffusion"),
    ("3DGS_diffusion", "EgoSurg"),
)
METRICS = ("PSNR", "SSIM", "LPIPS", "Flicker")
DEFAULT_SURGERIES = ["1", "2", "3", "4", "5", "6"]
DEFAULT_PHASES = ["01", "02", "03", "04", "05"]


def paper_precision(metric: str, value: float) -> str:
    if metric == "PSNR":
        return f"{value:.1f}"
    return f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--surgeries", nargs="+", default=DEFAULT_SURGERIES)
    parser.add_argument("--phases", nargs="+", default=DEFAULT_PHASES)
    args = parser.parse_args()

    values = {key: {metric: [] for metric in METRICS} for key, _ in METHODS}
    missing = []
    for surgery in args.surgeries:
        for phase in args.phases:
            path = (
                args.dataset_dir
                / surgery
                / f"{phase}_01"
                / "temporal_evaluation"
                / "summary_mean.csv"
            )
            if not path.is_file():
                missing.append(str(path))
                continue
            with path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    key = row["method"].strip()
                    if key not in values:
                        continue
                    for metric in METRICS:
                        values[key][metric].append(float(row[metric]))

    if missing:
        print(f"Missing {len(missing)} summary_mean.csv files.", file=sys.stderr)
        for path in missing:
            print(path, file=sys.stderr)
        sys.exit(1)

    print("method,n,PSNR,SSIM,LPIPS,Flicker")
    for key, label in METHODS:
        n = len(values[key]["PSNR"])
        if n < 2:
            print(f"{label} has {n} clips; need at least 2.", file=sys.stderr)
            sys.exit(1)
        cells = []
        precise = []
        for metric in METRICS:
            series = values[key][metric]
            mu, sigma = mean(series), stdev(series)
            cells.append(f"{mu:.4f} ± {sigma:.4f}")
            precise.append(
                f"{paper_precision(metric, mu)} ± {paper_precision(metric, sigma)}"
            )
        print(f"{label},{n}," + ",".join(cells))
        print(f"  Table 3B precision: {' | '.join(precise)}")


if __name__ == "__main__":
    main()
