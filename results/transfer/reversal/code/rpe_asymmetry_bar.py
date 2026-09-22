#!/usr/bin/env python
"""Simple bar plot: mean early-reversal RPE for the two stimuli that swap value at
reversal (100%->0%, stim index 2 = "newly worthless"; 0%->100%, stim index 0 = "newly
valuable"), one bar per model class per stimulus. Uses genuine live probe_rpe from
history.json (cxval.vigour.infer_rpe wired into make_probe) -- NOT available for runs
trained before that wiring was added (e.g. reversal_5000; see recovery_time.py's
rpe_vs_recovery_time docstring).

Unlike recovery_time.py's rpe_vs_recovery_time[_per_stim], this does NOT gate on
whether the seed went on to recover -- every seed with a genuine probe_rpe trace is
included, so this is a plain "what does the early RPE look like" summary, meant to be
looked at ALONGSIDE the causal reward-scale-intervention results (run_reward_scale_
intervention.sh) and the model-specific relearning-speed asymmetry: does the model
class that shows the biggest naturally-occurring RPE asymmetry between these two
stimuli also show the biggest relearning-speed asymmetry?

Produces two companion figures: magnitude (mean |RPE|, matching the convention used
elsewhere in this codebase for "does RPE size predict recovery speed") and signed
(mean RPE, sign and all -- the README's own description of the real RPE signature is
"a sharp NEGATIVE RPE dip for the newly-devalued stimulus... POSITIVE RPE bump for the
newly-valued stimulus", which the magnitude-only version can't show).

Usage:
  cd results/transfer/reversal
  python3 code/rpe_asymmetry_bar.py \
      --post-runs results/action_std_0p15_full/model_runs_reversal \
      --out results/action_std_0p15_full/figures_seed_groups
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import seed_groups as SG
except ModuleNotFoundError:
    import importlib
    SG = importlib.import_module("16_06_26_seed_groups")
F = SG.RA.F

STIM_LABELS = {0: "0%→100%\n(newly valuable)", 2: "100%→0%\n(newly worthless)"}


def early_rpe_by_model(post_runs, stim, window_trials=500):
    """{model_type: [(seed, mean_abs_rpe, mean_signed_rpe)]}, ALL seeds with a genuine
    probe_rpe trace in history.json -- no recovery-gating (see module docstring)."""
    out = {}
    for f in glob.glob(str(Path(post_runs) / "*" / "seed*" / "history.json")):
        p = Path(f)
        mt = p.parent.parent.name
        seed = int(p.parent.name[4:])
        try:
            h = json.loads(p.read_text())
        except Exception:
            continue
        pu, rpe = h.get("probe_update"), h.get("probe_rpe")
        if not pu or not rpe:
            continue
        tot = max(h.get("total_updates", len(pu)), 1)
        ntr = h.get("n_trials", 2500)
        trials = np.asarray(pu, float) * ntr / tot
        rpe = np.asarray(rpe, float)   # (n_probes, 3)
        mask = trials <= window_trials
        if not mask.any():
            mask = np.zeros(len(trials), bool)
            mask[0] = True
        mean_abs = float(np.abs(rpe[mask, stim]).mean())
        mean_signed = float(rpe[mask, stim].mean())
        out.setdefault(mt, []).append((seed, mean_abs, mean_signed))
    return out


def fig_rpe_asymmetry_bar(post_runs, out, stims=(2, 0), window_trials=500, signed=False,
                          fname="rpe_asymmetry_bar"):
    types = list(F.MODELS)
    data = {stim: early_rpe_by_model(post_runs, stim, window_trials) for stim in stims}

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    n_groups = len(stims)
    n_bars = len(types)
    width = 0.8 / n_bars
    x = np.arange(n_groups)
    for i, mt in enumerate(types):
        means, sems, ns = [], [], []
        for stim in stims:
            rows = data[stim].get(mt, [])
            vals = np.array([r[2] if signed else r[1] for r in rows])
            means.append(vals.mean() if len(vals) else np.nan)
            sems.append(vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
            ns.append(len(vals))
        offs = (i - (n_bars - 1) / 2) * width
        ax.bar(x + offs, means, width=width * 0.92, yerr=sems, capsize=4,
              color=F.MODELS[mt]["color"],
              label=f"{F.MODELS[mt]['label']} (n={ns[0] if ns else 0})")
    if signed:
        ax.axhline(0, color="0.5", lw=1, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([STIM_LABELS.get(s, str(s)) for s in stims], fontsize=10)
    kind = "signed" if signed else "|magnitude|"
    ax.set_ylabel(f"mean {kind} RPE, first {window_trials} reversal trials", fontsize=12)
    ax.set_title("Early post-reversal RPE by stimulus and model class\n"
                 "(all seeds, not gated on recovery)", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    F.save_fig(fig, out, fname + ("_signed" if signed else ""))

    print(f"\n  Early RPE ({kind}), first {window_trials} trials post-reversal:")
    for stim in stims:
        label = STIM_LABELS.get(stim, str(stim)).splitlines()[0]
        print(f"  stim {label}:")
        for mt in types:
            rows = data[stim].get(mt, [])
            vals = np.array([r[2] if signed else r[1] for r in rows])
            if len(vals):
                sem = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
                print(f"    {F.MODELS[mt]['label']:32s} n={len(vals):2d}  "
                      f"mean={vals.mean():+.4f}  sem={sem:.4f}")
            else:
                print(f"    {F.MODELS[mt]['label']:32s} n=0 (no genuine probe_rpe found)")
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--post-runs", default="action_std_0p15_full/model_runs_reversal")
    ap.add_argument("--out", default="action_std_0p15_full/figures_seed_groups")
    ap.add_argument("--window-trials", type=int, default=500)
    ap.add_argument("--style", default=str(F.DEFAULT_STYLE))
    args = ap.parse_args()
    matplotlib.use("Agg")
    if Path(args.style).exists():
        plt.style.use(args.style)
        print(f"style: {args.style}")
    else:
        print(f"** WARNING: style file not found at {args.style} -- default styling **")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_rpe_asymmetry_bar(args.post_runs, out, window_trials=args.window_trials, signed=False)
    fig_rpe_asymmetry_bar(args.post_runs, out, window_trials=args.window_trials, signed=True)
    print("\nWrote rpe_asymmetry_bar.png (+_signed variant) to", out)


if __name__ == "__main__":
    main()
