#!/usr/bin/env python
"""
Regenerate architecture sweep figures from a previously saved CSV.

Usage:
    python scripts/plot_arch_sweep.py results/arch_sweep_20260512_214006.csv
    python scripts/plot_arch_sweep.py results/arch_sweep_20260512_214006.csv --out-dir figures/
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import importlib.util

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# load plotting code and constants from the architecture sweep script by path
# (direct import isn't possible because the filename starts with a digit)
_arch_spec = importlib.util.spec_from_file_location(
    "sweep_model_architecture",
    Path(__file__).parent / "13_05_26_sweep_model_architecture.py",
)
_arch_mod = importlib.util.module_from_spec(_arch_spec)
_arch_spec.loader.exec_module(_arch_mod)
plot_results           = _arch_mod.plot_results
TRIALS_PER_HIDDEN      = _arch_mod.TRIALS_PER_HIDDEN
TRIALS_PER_HIDDEN_DEFAULT = _arch_mod.TRIALS_PER_HIDDEN_DEFAULT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Path to arch_sweep_*.csv")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: same directory as CSV)")
    parser.add_argument("--tag", default=None,
                        help="Timestamp tag for output filenames (default: derived from CSV name)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        sys.exit(1)

    # keep_default_na=False so "None" strings aren't silently converted to NaN
    df = pd.read_csv(csv_path, keep_default_na=False, na_values=[""])

    # normalise rank_key: old CSVs stored "None" / "1" as strings;
    # new CSVs store -1 / 1 as integers.
    def _to_rank_key(v):
        if v in ("None", "", None):
            return -1
        try:
            return int(float(v))
        except (ValueError, TypeError):
            return -1

    df["rank_key"] = df["rank_key"].map(_to_rank_key)

    hidden_sizes = sorted(int(h) for h in df["hidden_size"].unique())
    ranks_all    = sorted(int(r) for r in df["rank_key"].unique())
    n_seeds = int(df.groupby(["hidden_size", "rank_key"])["seed"].nunique().max())

    out_dir = Path(args.out_dir) if args.out_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # use timestamp from filename if possible, otherwise generate a new one
    if args.tag:
        ts = args.tag
    else:
        stem = csv_path.stem  # e.g. "arch_sweep_20260512_214006"
        parts = stem.split("_", 2)
        ts = parts[2] if len(parts) == 3 else datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"CSV      : {csv_path}")
    print(f"Combos   : {len(df.groupby(['hidden_size', 'rank_key']))}  "
          f"({len(hidden_sizes)} hidden sizes × {len(ranks_all)} rank configs)")
    print(f"Seeds    : up to {n_seeds} per combo")
    print(f"Out dir  : {out_dir}\n")

    plot_results(df, hidden_sizes, ranks_all, out_dir, ts, n_seeds)


if __name__ == "__main__":
    main()
