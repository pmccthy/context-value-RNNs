#!/usr/bin/env python
"""Rebuild figure_data.pkl + metrics.csv + activations.npz directly from the stored
``time_resolved/*.npz`` files — NO model, NO inference, NO cxval, NO PyTorch.

This is the SAME computation as reproducibility/inference/run_inference.py (vigour,
population activity, responder counts/fractions, per-unit tuning, trial-aligned means,
responsiveness), but sourced from the saved per-trial activity instead of a live model
rollout. So you can regenerate everything the figures need WITHOUT the training repo —
only numpy, scipy and pandas. The numbers match run_inference exactly (same arrays,
same t-test).

Usage (from the bundle root):
  python code/build_figure_data_from_timeresolved.py \
         --time-resolved figure_data/time_resolved --out figure_data
"""
from __future__ import annotations
import argparse, glob, json, pickle, re
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests

STIM_LABELS = ["0%", "50%", "100%"]
SEGMENTS = ["ITI", "stim", "outcome"]
METRIC_DESC = {
    "vigour":          "deterministic mean vigour per stimulus, in [0,1]",
    "pop_activity":    "mean hidden-unit activation in the stim window",
    "n_responsive":    "# units significantly responsive (paired t-test vs pre-stim baseline)",
    "frac_responsive": "fraction of units significantly responsive",
}
_ORDER = {"classif_rl": 0, "rl_only": 1, "classif_rl_readout_only": 2}


def per_model(z, direction="excitatory", alpha=0.05, correction="fdr_bh"):
    """Compute one model's metrics + tuning + aligned means + responsiveness from its
    stored per-trial activity. Mirrors run_inference / responsive_proportions_ttest.

    direction: which units count as responsive to a stimulus —
      "excitatory" (default, = run_inference): response significantly ABOVE baseline;
      "two_sided": ANY significant difference (incl. suppressed); "suppressed": below.

    correction: multiple-comparisons correction applied ACROSS the H units
    being tested simultaneously for each stimulus (the classic "testing
    thousands of neurons/units" multiple-testing problem -- previously
    absent here, a flat p<alpha was used per unit). "fdr_bh" (default)
    Benjamini-Hochberg FDR via statsmodels; None reproduces the old,
    uncorrected behaviour (kept for comparison / the alpha-sweep analysis)."""
    aligned = np.asarray(z["aligned"], np.float32)         # (trial, time, unit)
    stim = np.asarray(z["stimulus"]); vig = np.asarray(z["vigour"], float)
    n_iti, stim_ts, rew_ts = (int(b) for b in z["bounds"])
    H = aligned.shape[1 + 1]                                # unit axis
    R = aligned[:, n_iti:n_iti + stim_ts, :].mean(1)        # (trial, unit) stim-window mean
    Bl = aligned[:, 0:n_iti, :].mean(1)                     # (trial, unit) baseline mean
    delta = R - Bl
    tuning = np.stack([R[stim == s].mean(0) for s in range(3)])              # (3, H)
    aligned_mean = np.stack([aligned[stim == s].mean(0) for s in range(3)])  # (3, n_align, H)
    resp = np.zeros((H, 3), bool)
    sc = {m: np.zeros(3) for m in METRIC_DESC}
    for s in range(3):
        idx = stim == s
        sc["vigour"][s] = vig[idx].mean()
        sc["pop_activity"][s] = R[idx].mean()              # == tuning[s].mean()
        t, p = ttest_1samp(delta[idx], 0.0, axis=0)        # paired (R-Bl) vs 0, per unit
        if direction == "excitatory":                      # response significantly ABOVE baseline
            p_one = np.where(t > 0, p / 2, 1.0)             # wrong-direction units forced non-sig
        elif direction == "suppressed":                    # significantly BELOW baseline
            p_one = np.where(t < 0, p / 2, 1.0)
        else:                                              # "two_sided": ANY significant diff
            p_one = p
        p_one = np.nan_to_num(p_one, nan=1.0)
        if correction is not None:
            # FDR-BH across the H units tested for this stimulus -- the family
            # is fixed at H regardless of direction (wrong-direction units,
            # forced to p=1 above, correctly count toward the family).
            _, p_use, _, _ = multipletests(p_one, alpha=alpha, method=correction)
        else:
            p_use = p_one
        sig = np.nan_to_num(p_use < alpha).astype(bool)
        resp[:, s] = sig
        sc["n_responsive"][s] = int(sig.sum())
        sc["frac_responsive"][s] = float(sig.mean())
    return dict(scalars=sc, tuning=tuning.astype(np.float32),
                aligned_mean=aligned_mean.astype(np.float32), responsive=resp,
                bounds=(n_iti, stim_ts, rew_ts), n_align=aligned.shape[1])


def build_data(time_resolved_dir, direction="excitatory", alpha=0.05, correction="fdr_bh"):
    """Build the full figure_data dict from the time_resolved/*.npz files, under the
    chosen responder `direction`. Importable (used by the figures, the notebook and the
    responder-definition comparison) — returns the same dict run_inference writes."""
    results = {}
    for f in sorted(glob.glob(str(Path(time_resolved_dir) / "*.npz"))):
        m = re.match(r"(.+)_seed(\d+)\.npz", Path(f).name)
        if not m:
            continue
        z = np.load(f, allow_pickle=True)
        results[(m.group(1), int(m.group(2)))] = per_model(z, direction=direction, alpha=alpha,
                                                            correction=correction)
    if not results:
        raise FileNotFoundError(f"no .npz files in {time_resolved_dir}")

    types = sorted({mt for mt, _ in results}, key=lambda m: (_ORDER.get(m, 9), m))
    seeds = sorted({s for _, s in results})
    ti = {m: i for i, m in enumerate(types)}; si = {s: i for i, s in enumerate(seeds)}
    A, S, K = len(types), len(seeds), 3
    any_r = next(iter(results.values())); H = any_r["tuning"].shape[1]
    n_align = any_r["n_align"]; bounds = any_r["bounds"]

    scal = {m: np.full((A, S, K), np.nan) for m in METRIC_DESC}
    tuning = np.full((A, S, K, H), np.nan, np.float32)
    aligned = np.full((A, S, H, n_align, K), np.nan, np.float32)
    responsive = np.zeros((A, S, H, K), bool)
    rows = []
    for (mt, seed), r in results.items():
        a, s = ti[mt], si[seed]
        for m in METRIC_DESC:
            scal[m][a, s] = r["scalars"][m]
        tuning[a, s] = r["tuning"]
        aligned[a, s] = np.transpose(r["aligned_mean"], (2, 1, 0))   # (unit, time, stim)
        responsive[a, s] = r["responsive"]
        for k in range(3):
            rows.append(dict(model_type=mt, seed=seed, stim=k, stim_label=STIM_LABELS[k],
                             vigour=r["scalars"]["vigour"][k],
                             pop_activity=r["scalars"]["pop_activity"][k],
                             n_responsive=int(r["scalars"]["n_responsive"][k]),
                             frac_responsive=r["scalars"]["frac_responsive"][k], n_units=H))

    coords = {"model_type": types, "seed": seeds, "stim": STIM_LABELS}
    n_iti, stim_ts, rew_ts = bounds
    data = {
        "description": "Rebuilt from time_resolved/ (no inference). Per-(model_type, seed, "
                       "stimulus) metrics, per-unit stim tuning, per-unit trial-aligned means, "
                       "and responsiveness. Backs every figure.",
        "model_types": types, "seeds": seeds, "stim_labels": STIM_LABELS, "n_units": H,
        "seeds_present": {m: sorted({s for mm, s in results if mm == m}) for m in types},
        "period": {"n_iti_pre": n_iti, "stim_ts": stim_ts, "rew_ts": rew_ts,
                   "n_align_ts": n_align, "segments": SEGMENTS,
                   "segment_bounds": [[0, n_iti], [n_iti, n_iti + stim_ts],
                                      [n_iti + stim_ts, n_align]]},
        "scalars": {"dims": ["model_type", "seed", "stim"], "coords": coords,
                    "units": METRIC_DESC, **scal},
        "tuning": {"description": "per-unit mean stim-window activation",
                   "dims": ["model_type", "seed", "stim", "unit"],
                   "coords": {**coords, "unit": list(range(H))}, "data": tuning},
        "aligned_mean": {"description": "per-unit trial-aligned mean activation per stimulus",
                         "dims": ["model_type", "seed", "unit", "time", "stim"],
                         "coords": {"model_type": types, "seed": seeds, "unit": list(range(H)),
                                    "time": list(range(n_align)), "stim": STIM_LABELS},
                         "data": aligned},
        "responsive": {"description": "unit significantly responds to stimulus (defines groups)",
                       "dims": ["model_type", "seed", "unit", "stim"],
                       "coords": {"model_type": types, "seed": seeds, "unit": list(range(H)),
                                  "stim": STIM_LABELS}, "data": responsive},
    }
    return data, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-resolved", default="figure_data/time_resolved")
    ap.add_argument("--out", default="figure_data")
    ap.add_argument("--direction", default="excitatory",
                    choices=["excitatory", "two_sided", "suppressed"],
                    help="responder rule (default excitatory = same as run_inference)")
    ap.add_argument("--correction", default="fdr_bh",
                    help="multiple-comparisons correction across units per stimulus "
                         "(statsmodels multipletests method name, e.g. fdr_bh, "
                         "bonferroni); pass 'none' to disable (old, uncorrected "
                         "behaviour, kept for comparison)")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    correction = None if args.correction.lower() == "none" else args.correction
    data, results = build_data(args.time_resolved, direction=args.direction, correction=correction)
    rows = [dict(model_type=mt, seed=seed, stim=k, stim_label=STIM_LABELS[k],
                 vigour=r["scalars"]["vigour"][k], pop_activity=r["scalars"]["pop_activity"][k],
                 n_responsive=int(r["scalars"]["n_responsive"][k]),
                 frac_responsive=r["scalars"]["frac_responsive"][k], n_units=data["n_units"])
            for (mt, seed), r in results.items() for k in range(3)]
    pickle.dump(data, open(out / "figure_data.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    np.savez(out / "activations.npz", **{f"{mt}_{s}": r["tuning"] for (mt, s), r in results.items()})
    print(f"rebuilt from {len(results)} time_resolved files (direction={args.direction}) "
          f"| types={data['model_types']} | H={data['n_units']}")
    for w in ("figure_data.pkl", "metrics.csv", "activations.npz"):
        print(f"  -> {out / w}")


if __name__ == "__main__":
    main()
