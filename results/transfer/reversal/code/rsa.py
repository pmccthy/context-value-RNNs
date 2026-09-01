#!/usr/bin/env python
"""RSA (representational similarity analysis): per-model representational dissimilarity
matrices (RDMs), plus a second-order comparison of how similar the three models'
representational GEOMETRIES are to each other. Two condition sets ("first-order" RSAs):

  TIME-RESOLVED  conditions = the 3 stimuli x the n_align trial-time bins (ITI -> stim ->
                 outcome), i.e. each condition is "this stimulus, at this point in the
                 trial" (33 conditions). Uses aligned_mean (figures.py's _pool_units).
  STIMULUS-ONLY  conditions = just the 3 stimuli, using each unit's AVERAGE stim-window
                 response (the same tuning data behind the tuning heatmaps elsewhere in
                 this bundle) — no trial-time dimension. Coarser (3 conditions) but the
                 more standard/simple RSA and the one most directly comparable to typical
                 single-timepoint neural tuning data.

For either, a model's population-activity vector per condition is built by pooling units
over all its seeds, then

    RDM[i, j] = 1 - Pearson_r( population_vector_i, population_vector_j )

correlating across the UNIT axis (not neuron-by-neuron, not over trials) — the standard
condition x condition correlation-distance RDM. 0 = identical population pattern,
2 = perfectly anti-correlated, 1 = uncorrelated.

  -> rsa_rdm_heatmaps            time-resolved RDM per model (3 blocks of n_align),
                                 shared colour scale, block/segment boundaries marked
  -> rsa_second_order            model x model heatmap: Spearman correlation between
                                 each pair of models' time-resolved RDMs (upper
                                 triangles) — geometry similarity, independent of scale
  -> rsa_rdm_heatmaps_stim       stimulus-only (3x3) RDM per model, values annotated
  -> rsa_second_order_stim       same second-order comparison for the stimulus-only
                                 RDMs (only 3 conditions -> 3 upper-triangle entries, so
                                 treat this one as a rough/low-powered read)

Usage:
  python scripts/16_06_26_rsa.py --data results/figure_data --out results/figures_rsa
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, matplotlib; import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import figures as F
except ModuleNotFoundError:
    import importlib
    F = importlib.import_module("16_06_26_figures")


def _population_rdm(D, model_type):
    """Condition x condition correlation-distance RDM for one model, conditions ordered
    stim-major (block 0 = all n_align timepoints of stim 0, etc). Returns (rdm (C,C),
    n_units) or (None, None) if the model has no data."""
    pats, traces = F._pool_units(D, model_type)              # traces: (N units, n_align, 3 stim)
    if traces is None:
        return None, None
    n_units, n_align, n_stim = traces.shape
    # condition-by-unit matrix, stim-major order: row c = (stim s, time t) -> traces[:, t, s]
    cond = traces.transpose(2, 1, 0).reshape(n_stim * n_align, n_units)   # (C, N)
    with np.errstate(invalid="ignore"):
        R = np.corrcoef(cond)
    rdm = 1.0 - R
    return rdm, n_units


def _pool_units_tuning(D, model_type):
    """Pool each unit's AVERAGE stim-window response (D['tuning'], not the time-resolved
    trace) over a model's seeds. Returns (N units, 3 stim), or None if absent."""
    ti = D["model_types"].index(model_type)
    seed_idx = [D["seeds"].index(s) for s in D["seeds_present"].get(model_type, [])]
    if not seed_idx:
        return None
    mats = [D["tuning"]["data"][ti, si].T for si in seed_idx]     # each (unit, 3 stim)
    return np.concatenate(mats, 0)


def _population_rdm_stim(D, model_type):
    """Stimulus-only (3x3) correlation-distance RDM for one model — the simpler,
    'first-order RSA' companion to _population_rdm, using each unit's AVERAGE
    stim-window response instead of the time-resolved trial trace. Returns
    (rdm (3,3), n_units) or (None, None)."""
    pooled = _pool_units_tuning(D, model_type)                    # (N, 3)
    if pooled is None:
        return None, None
    cond = pooled.T                                               # (3 stim, N)
    with np.errstate(invalid="ignore"):
        R = np.corrcoef(cond)
    return 1.0 - R, pooled.shape[0]


def _condition_ticks(D):
    """Tick positions/labels + block & segment boundaries for the stim-major condition axis."""
    n_iti, stim_ts, rew_ts = (D["period"][k] for k in ("n_iti_pre", "stim_ts", "rew_ts"))
    n_align = n_iti + stim_ts + rew_ts
    stim = D["stim_labels"]
    block_bounds = [b * n_align for b in range(len(stim) + 1)]           # outer stim blocks
    seg_bounds = []                                                      # ITI|stim, stim|outcome, per block
    for b in range(len(stim)):
        base = b * n_align
        seg_bounds += [base + n_iti, base + n_iti + stim_ts]
    tick_pos = [b * n_align + n_align / 2 - 0.5 for b in range(len(stim))]
    return n_align, tick_pos, stim, block_bounds, seg_bounds


def fig_rdm_heatmaps(D, out, model_types=None, vmax=None):
    """One condition x condition RDM heatmap per model (shared colour scale), conditions
    = stimulus (outer, block-ordered) x trial-time (inner). Thick lines mark stimulus
    blocks, thin dotted lines the ITI|stim|outcome boundaries within each block."""
    mts = [m for m in F.MODELS if m in D["model_types"]] if model_types is None else \
          [m for m in F.MODELS if m in model_types]
    rdms = {}
    for m in mts:
        rdm, n_units = _population_rdm(D, m)
        if rdm is not None:
            rdms[m] = (rdm, n_units)
    if not rdms:
        print("  (skip rsa_rdm_heatmaps: no models with data)"); return rdms
    n_align, ticks, stim, block_bounds, seg_bounds = _condition_ticks(D)
    if vmax is None:
        vmax = max(np.nanpercentile(r, 99) for r, _ in rdms.values())
    types = [m for m in mts if m in rdms]
    fig, axes = plt.subplots(1, len(types), figsize=(4.6 * len(types) + 1.0, 5.0), squeeze=False)
    im = None
    for ax, mt in zip(axes[0], types):
        rdm, n_units = rdms[mt]
        im = ax.imshow(rdm, cmap="viridis", vmin=0, vmax=vmax, interpolation="nearest")
        for b in block_bounds[1:-1]:
            ax.axvline(b - 0.5, color="white", lw=1.6)
            ax.axhline(b - 0.5, color="white", lw=1.6)
        for b in seg_bounds:
            ax.axvline(b - 0.5, color="white", lw=0.6, ls=":")
            ax.axhline(b - 0.5, color="white", lw=0.6, ls=":")
        ax.set_xticks(ticks); ax.set_xticklabels(stim, fontsize=11)
        ax.set_yticks(ticks); ax.set_yticklabels(stim, fontsize=11)
        ax.set_title(f"{F.MODELS[mt]['label']}\n(n={n_units} units, pooled)", fontsize=11)
    fig.colorbar(im, ax=axes, shrink=0.75, pad=0.02, label="dissimilarity (1 − r)")
    fig.suptitle("Representational dissimilarity matrices — condition = stimulus × trial-time\n"
                 "thick line = stimulus block, dotted = ITI|stim|outcome", fontsize=12)
    F.save_fig(fig, out, "rsa_rdm_heatmaps")
    return rdms


def fig_rdm_heatmaps_stim(D, out, model_types=None, vmax=None):
    """One condition x condition RDM heatmap per model, conditions = the 3 stimuli only
    (each unit's AVERAGE stim-window response, not the trial-time trace) — the simpler
    'first-order RSA' companion to fig_rdm_heatmaps' time-resolved version. Values are
    annotated since there are only 9 cells per panel."""
    mts = [m for m in F.MODELS if m in D["model_types"]] if model_types is None else \
          [m for m in F.MODELS if m in model_types]
    rdms = {}
    for m in mts:
        rdm, n_units = _population_rdm_stim(D, m)
        if rdm is not None:
            rdms[m] = (rdm, n_units)
    if not rdms:
        print("  (skip rsa_rdm_heatmaps_stim: no models with data)"); return rdms
    stim = D["stim_labels"]
    if vmax is None:
        vmax = max(np.nanpercentile(r, 99) for r, _ in rdms.values())
    types = [m for m in mts if m in rdms]
    fig, axes = plt.subplots(1, len(types), figsize=(3.4 * len(types) + 1.0, 4.0), squeeze=False)
    im = None
    for ax, mt in zip(axes[0], types):
        rdm, n_units = rdms[mt]
        im = ax.imshow(rdm, cmap="viridis", vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_xticks(range(3)); ax.set_xticklabels(stim, fontsize=11)
        ax.set_yticks(range(3)); ax.set_yticklabels(stim, fontsize=11)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{rdm[i, j]:.2f}", ha="center", va="center", fontsize=11,
                        color="w" if rdm[i, j] < vmax * 0.6 else "k")
        ax.set_title(f"{F.MODELS[mt]['label']}\n(n={n_units} units, pooled)", fontsize=11)
    fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02, label="dissimilarity (1 − r)")
    fig.suptitle("Representational dissimilarity matrices — condition = stimulus only\n"
                 "(average stim-window response, not time-resolved)", fontsize=12)
    F.save_fig(fig, out, "rsa_rdm_heatmaps_stim")
    return rdms


def fig_second_order(rdms, out, stem="rsa_second_order", note=""):
    """Model x model heatmap: Spearman correlation between models' RDM upper-triangles —
    similarity of representational GEOMETRY (independent of any model's activity scale).
    `rdms` = a dict returned by fig_rdm_heatmaps / fig_rdm_heatmaps_stim."""
    types = [m for m in F.MODELS if m in rdms]
    if len(types) < 2:
        print(f"  (skip {stem}: need >=2 models)"); return
    iu = np.triu_indices_from(rdms[types[0]][0], k=1)
    vecs = {m: rdms[m][0][iu] for m in types}
    n = len(types); S = np.eye(n)
    for i, a in enumerate(types):
        for j, b in enumerate(types):
            if i < j:
                r, _ = spearmanr(vecs[a], vecs[b])
                S[i, j] = S[j, i] = r
    fig, ax = plt.subplots(figsize=(1.7 * n + 2.2, 1.7 * n + 1.4))
    im = ax.imshow(S, cmap="magma", vmin=0, vmax=1)
    labels = [F.MODELS[m]["label"] for m in types]
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=10)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center", fontsize=12,
                    color="w" if S[i, j] < 0.6 else "k")
    fig.colorbar(im, ax=ax, shrink=0.85, label="RDM correlation (Spearman)")
    ax.set_title("Second-order RSA: similarity of representational\n"
                 f"geometry between models{note}", fontsize=12)
    fig.tight_layout(); F.save_fig(fig, out, stem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/figure_data")
    ap.add_argument("--out", default="figures_rsa")
    ap.add_argument("--style", default=str(F.DEFAULT_STYLE))
    args = ap.parse_args()
    matplotlib.use("Agg")
    if Path(args.style).exists():
        plt.style.use(args.style)
    else:
        print(f"** WARNING: style file not found at {args.style} — figures will use "
              f"default matplotlib styling, not the house style **")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    D = F.load(args.data)
    rdms = fig_rdm_heatmaps(D, out)
    fig_second_order(rdms, out)
    rdms_stim = fig_rdm_heatmaps_stim(D, out)
    fig_second_order(rdms_stim, out, stem="rsa_second_order_stim",
                     note=" (stimulus-only,\nonly 3 conditions — low-powered)")


if __name__ == "__main__":
    main()
