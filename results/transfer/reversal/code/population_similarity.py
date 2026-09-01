#!/usr/bin/env python
"""Population-level representation comparisons: per-neuron pre/post reversal change,
and a responder-group chi-squared goodness-of-fit test against (placeholder, for now)
neural data — plus a "which model fits best" ranking.

Two independent analyses:

  1. Per-neuron pre/post reversal change (needs --pre AND --post figure_data):
     for every hidden unit, the Pearson correlation between its pre- and post-reversal
     tuning vector (its response to the 3 stimuli) and its mean signed activity
     difference (post - pre). High correlation + small difference = unit kept the same
     tuning shape and level; low/negative correlation = tuning reorganised.
       -> population_neuron_scatter        corr (x) vs diff (y), one panel per model
       -> population_neuron_distributions  histograms of corr and of diff, per model
       -> population_diff_summary          bar of mean |post-pre| per model

  2. Responder-group chi-squared fit to data (needs one figure_data, --group-data or
     --pre): each model's units are already classified into the 7 responder groups
     (0%-only, 50%-only, 100%-only, the 3 pairs, all-three; see figures.py GROUPS).
     Pool counts over seeds, rescale a REAL (or, until you have one, FAKE placeholder)
     dataset's group proportions to the model's N, and run a chi-squared goodness-of-fit
     test per model. The model with the lowest chi-squared statistic (highest p-value;
     dof is identical across models) is the representation LEAST distinguishable from
     the data on this criterion.
       -> population_group_chi2               bar of chi2 statistic per model, p-values
       -> population_group_proportions_vs_data grouped bars: each model's group
                                                composition next to the data's

Supply real neural-data group counts with --data-counts group_counts.json, formatted as
  {"0%-only": 41, "50%-only": 33, "100%-only": 52, "0% & 50%": 12,
   "0% & 100%": 9, "50% & 100%": 14, "all three": 22}
Until then, a clearly-labelled FAKE dataset (multinomial draw from an illustrative
proportion vector) stands in, so the pipeline and numbers are ready to go.

Usage:
  python scripts/16_06_26_population_similarity.py \
         --pre results/figure_data --post results/reversal_5000/figure_data_reversal \
         --out results/figures_population_similarity
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, matplotlib; import matplotlib.pyplot as plt
from scipy import stats as sstats

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import figures as F
except ModuleNotFoundError:
    import importlib
    F = importlib.import_module("16_06_26_figures")

GROUP_LABELS = [F.GROUPS[k] for k in F.GROUP_ORDER]   # canonical order used everywhere below


# ── 1. per-neuron pre/post reversal change ──────────────────────────────────────────
def per_neuron_pre_post(pre, post, model_type, responders_only=True):
    """Per-neuron pre/post tuning correlation (Pearson r across the 3 stimuli) and mean
    SIGNED activity difference (post - pre), pooled over units & matched seeds. Returns
    (corr (N,), diff (N,)), or (None, None) if the model isn't in both datasets."""
    if model_type not in pre["model_types"] or model_type not in post["model_types"]:
        return None, None
    tp, to = pre["model_types"].index(model_type), post["model_types"].index(model_type)
    seeds = sorted(set(pre["seeds_present"].get(model_type, [])) &
                    set(post["seeds_present"].get(model_type, [])))
    corrs, diffs = [], []
    for s in seeds:
        a = pre["tuning"]["data"][tp, pre["seeds"].index(s)]      # (3 stim, H units)
        b = post["tuning"]["data"][to, post["seeds"].index(s)]
        if responders_only:
            resp = pre["responsive"]["data"][tp, pre["seeds"].index(s)].any(1)   # (H,)
            a, b = a[:, resp], b[:, resp]
        ac, bc = a - a.mean(0), b - b.mean(0)
        denom = np.sqrt((ac ** 2).sum(0) * (bc ** 2).sum(0))
        c = np.where(denom > 1e-9, (ac * bc).sum(0) / denom.clip(1e-9), np.nan)
        d = (b - a).mean(0)                                        # avg over the 3 stimuli
        corrs.append(c); diffs.append(d)
    if not corrs:
        return None, None
    return np.concatenate(corrs), np.concatenate(diffs)


def print_neuron_summary(pre, post, responders_only):
    types = [m for m in F.MODELS if m in post["model_types"] and m in pre["model_types"]]
    tag = "pre-reversal responders only" if responders_only else "all units"
    print(f"\nper-neuron pre/post reversal change ({tag}):")
    for mt in types:
        c, d = per_neuron_pre_post(pre, post, mt, responders_only)
        if c is None:
            continue
        cc = c[~np.isnan(c)]
        print(f"  {F.MODELS[mt]['label']:32s} n={len(cc):4d}  "
              f"corr: mean={np.mean(cc):+.3f} median={np.median(cc):+.3f}  "
              f"|post-pre|: mean={np.mean(np.abs(d)):.3f}")


def fig_neuron_scatter(pre, post, out, responders_only=True):
    """Scatter of per-neuron pre/post tuning correlation vs. mean activity difference,
    one panel per model class, pooled over units & matched seeds."""
    types = [m for m in F.MODELS if m in post["model_types"] and m in pre["model_types"]]
    fig, axes = plt.subplots(1, len(types), figsize=(4.6 * len(types), 4.7),
                             sharex=True, sharey=True, squeeze=False)
    for ax, mt in zip(axes[0], types):
        c, d = per_neuron_pre_post(pre, post, mt, responders_only)
        if c is None:
            ax.axis("off"); continue
        ax.scatter(c, d, s=11, alpha=0.35, color=F.MODELS[mt]["color"], edgecolor="none")
        ax.axvline(0, color="0.7", lw=0.8); ax.axhline(0, color="0.7", lw=0.8)
        ax.set_xlim(-1.08, 1.08)
        n = int(np.sum(~np.isnan(c)))
        ax.set_title(f"{F.MODELS[mt]['label']}\n(n={n} units)", fontsize=11)
        ax.set_xlabel("pre/post tuning correlation", fontsize=13)
    axes[0][0].set_ylabel("mean activity difference (post − pre)", fontsize=13)
    tag = " · pre-reversal responders only" if responders_only else " · all units"
    fig.suptitle("Per-neuron representation change across reversal" + tag, fontsize=13)
    fig.tight_layout(); F.save_fig(fig, out, "population_neuron_scatter")


def fig_neuron_distributions(pre, post, out, responders_only=True):
    """Per model: histograms of the per-neuron pre/post correlation and of the mean
    activity difference (density-normalised so class sizes differ don't matter)."""
    types = [m for m in F.MODELS if m in post["model_types"] and m in pre["model_types"]]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7))
    bins_c = np.linspace(-1, 1, 31)
    all_d = []
    cache = {}
    for mt in types:
        c, d = per_neuron_pre_post(pre, post, mt, responders_only)
        cache[mt] = (c, d)
        if d is not None:
            all_d.append(d)
    bins_d = np.linspace(np.nanmin(np.concatenate(all_d)), np.nanmax(np.concatenate(all_d)), 31) \
        if all_d else 30
    for mt in types:
        c, d = cache[mt]
        if c is None:
            continue
        col = F.MODELS[mt]["color"]
        axes[0].hist(c[~np.isnan(c)], bins=bins_c, histtype="step", lw=2.2, color=col,
                     density=True, label=F.MODELS[mt]["label"])
        axes[1].hist(d[~np.isnan(d)], bins=bins_d, histtype="step", lw=2.2, color=col, density=True)
    axes[0].set_xlabel("pre/post tuning correlation", fontsize=13); axes[0].set_ylabel("density", fontsize=13)
    axes[0].axvline(0, color="0.6", lw=0.9); axes[0].legend(fontsize=9)
    axes[0].set_title("Tuning-shape preservation")
    axes[1].set_xlabel("mean activity difference (post − pre)", fontsize=13)
    axes[1].axvline(0, color="0.6", lw=0.9); axes[1].set_title("Activity-level shift")
    tag = " · pre-reversal responders only" if responders_only else " · all units"
    fig.suptitle("Per-neuron distributions" + tag, fontsize=13)
    fig.tight_layout(); F.save_fig(fig, out, "population_neuron_distributions")


def fig_diff_summary(pre, post, out, responders_only=True):
    """Per class: mean |post - pre| per-unit activity difference, mean ± SEM over
    matched seeds (companion to figures.py's fig_tuning_correlation, which covers the
    correlation half of this same comparison)."""
    types = [m for m in F.MODELS if m in post["model_types"] and m in pre["model_types"]]
    x = np.arange(len(types)); means, sems, cols = [], [], []
    for mt in types:
        tp, to = pre["model_types"].index(mt), post["model_types"].index(mt)
        seeds = sorted(set(pre["seeds_present"].get(mt, [])) & set(post["seeds_present"].get(mt, [])))
        per_seed = []
        for s in seeds:
            a = pre["tuning"]["data"][tp, pre["seeds"].index(s)]
            b = post["tuning"]["data"][to, post["seeds"].index(s)]
            if responders_only:
                resp = pre["responsive"]["data"][tp, pre["seeds"].index(s)].any(1)
                a, b = a[:, resp], b[:, resp]
            per_seed.append(np.nanmean(np.abs(b - a)))
        means.append(np.mean(per_seed)); sems.append(np.std(per_seed) / np.sqrt(max(len(per_seed), 1)))
        cols.append(F.MODELS[mt]["color"])
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.bar(x, means, 0.6, yerr=sems, capsize=4, color=cols, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels([F.MODELS[m]["label"] for m in types], rotation=20, ha="right")
    ax.tick_params(axis="x", labelsize=12)
    ax.set_ylabel("mean |post − pre| activity", fontsize=13); ax.set_ylim(0, None)
    ax.set_title("Activity-level shift across reversal (0 = unchanged)", fontsize=13)
    fig.tight_layout(); F.save_fig(fig, out, "population_diff_summary")


# ── 2. responder-group chi-squared fit to (placeholder) data ────────────────────────
def fake_data_counts(total=300, seed=0):
    """PLACEHOLDER stand-in for real neural-data responder-group counts, in GROUP_LABELS
    order, until real counts are available (--data-counts). An illustrative proportion
    vector, multinomial-sampled so it looks like real cell counts rather than exact
    fractions. NOT real data."""
    rng = np.random.default_rng(seed)
    p = np.array([0.18, 0.14, 0.20, 0.08, 0.06, 0.08, 0.10])   # matches GROUP_LABELS order
    p = p / p.sum()
    return rng.multinomial(total, p).astype(float)


def load_data_counts(path):
    """Real neural-data group counts from a JSON {group_label: count}, in GROUP_LABELS
    order. Group labels must match figures.py's GROUPS values exactly."""
    d = json.loads(Path(path).read_text())
    missing = [l for l in GROUP_LABELS if l not in d]
    if missing:
        raise ValueError(f"--data-counts is missing group(s): {missing}. Expected keys: {GROUP_LABELS}")
    return np.array([d[l] for l in GROUP_LABELS], float)


def group_counts_pooled(D, model_type):
    """Pooled (summed over seeds) responder-group counts for one model, GROUP_LABELS order."""
    c = F.responder_group_counts(D, model_type)     # (n_seed_present, 7)
    return c.sum(0) if c.size else np.zeros(len(F.GROUP_ORDER))


def chi2_fit_to_data(D, data_counts, model_types=None):
    """Chi-squared goodness-of-fit, per model, of its pooled responder-group counts to
    `data_counts`'s PROPORTIONS (rescaled to the model's own total N, so this compares
    composition/shape, not raw neuron count). Returns {model_type: {chi2, p, dof, obs, exp}}."""
    mts = [m for m in F.MODELS if m in D["model_types"]] if model_types is None else \
          [m for m in F.MODELS if m in model_types]
    data_p = np.asarray(data_counts, float); data_p = data_p / data_p.sum()
    out = {}
    for mt in mts:
        obs = group_counts_pooled(D, mt)
        if obs.sum() == 0:
            continue
        exp = data_p * obs.sum()
        chi2, p = sstats.chisquare(f_obs=obs, f_exp=exp)
        out[mt] = dict(chi2=float(chi2), p=float(p), dof=len(GROUP_LABELS) - 1, obs=obs, exp=exp)
    return out


def fig_group_chi2(D, data_counts, out, model_types=None):
    """Bar of the chi-squared statistic per model (lower = closer fit to data), with
    per-model p-value annotated and the best fit outlined. Returns (fit, best_model)."""
    fit = chi2_fit_to_data(D, data_counts, model_types)
    types = [m for m in F.MODELS if m in fit]
    if not types:
        print("  (skip population_group_chi2: no models with responder-group data)")
        return fit, None
    best = min(types, key=lambda m: fit[m]["chi2"])
    x = np.arange(len(types)); chi2s = [fit[m]["chi2"] for m in types]
    fig, ax = plt.subplots(figsize=(6.8, 5))
    ax.bar(x, chi2s, 0.6, color=[F.MODELS[m]["color"] for m in types],
           edgecolor=["k" if m == best else "white" for m in types],
           linewidth=[2.4 if m == best else 1 for m in types])
    for xi, m in zip(x, types):
        ax.text(xi, fit[m]["chi2"] + max(chi2s) * 0.02, f"p={fit[m]['p']:.3f}",
                ha="center", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels([F.MODELS[m]["label"] for m in types], rotation=20, ha="right")
    ax.tick_params(axis="x", labelsize=12)
    ax.set_ylabel(r"$\chi^2$ statistic (goodness-of-fit vs. data)", fontsize=13)
    ax.set_title(f"Which representation best matches the data?  (dof={fit[types[0]]['dof']})\n"
                 f"lower = closer fit  ·  best: {F.MODELS[best]['label']}", fontsize=12)
    fig.tight_layout(); F.save_fig(fig, out, "population_group_chi2")
    return fit, best


def fig_group_proportions_vs_data(D, data_counts, out, model_types=None):
    """Grouped bars: % of responsive units in each responder group, one bar per model
    plus a 'data' bar, for a visual read alongside the chi-squared numbers."""
    mts = [m for m in F.MODELS if m in D["model_types"]] if model_types is None else \
          [m for m in F.MODELS if m in model_types]
    labels = GROUP_LABELS; x = np.arange(len(labels))
    n_series = len(mts) + 1; w = 0.8 / n_series
    fig, ax = plt.subplots(figsize=(12, 5.4))
    data_arr = np.asarray(data_counts, float); data_frac = 100 * data_arr / data_arr.sum()
    for i, m in enumerate(mts):
        c = group_counts_pooled(D, m)
        frac = 100 * c / max(c.sum(), 1e-9)
        ax.bar(x + (i - n_series / 2 + 0.5) * w, frac, w, color=F.MODELS[m]["color"],
               edgecolor="white", label=F.MODELS[m]["label"])
    ax.bar(x + (len(mts) - n_series / 2 + 0.5) * w, data_frac, w, color="0.25",
           edgecolor="white", label="data", hatch="//")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.tick_params(axis="x", labelsize=12)
    ax.set_ylabel("% of responsive units")
    ax.set_title("Responder-group composition: models vs. data")
    ax.legend(fontsize=10)
    fig.tight_layout(); F.save_fig(fig, out, "population_group_proportions_vs_data")


# ── main ──────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", default="results/figure_data")
    ap.add_argument("--post", default=None,
                    help="post-reversal figure_data, for the per-neuron pre/post plots "
                         "(omit to skip that half and only run the chi-squared fit)")
    ap.add_argument("--group-data", default=None,
                    help="figure_data to use for the responder-group chi-squared test "
                         "(default: --pre)")
    ap.add_argument("--data-counts", default=None,
                    help="JSON {group_label: count} of REAL neural-data responder-group "
                         "counts. Omitted -> a clearly-labelled FAKE placeholder dataset is used.")
    ap.add_argument("--fake-total", type=int, default=300)
    ap.add_argument("--fake-seed", type=int, default=0)
    ap.add_argument("--all-units", action="store_true",
                    help="use ALL units for the per-neuron pre/post plots, not just "
                         "pre-reversal responders")
    ap.add_argument("--out", default="figures_population_similarity")
    ap.add_argument("--style", default=str(F.DEFAULT_STYLE))
    args = ap.parse_args()
    matplotlib.use("Agg")
    if Path(args.style).exists():
        plt.style.use(args.style)
    else:
        print(f"** WARNING: style file not found at {args.style} — figures will use "
              f"default matplotlib styling, not the house style **")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    if args.post:
        pre, post = F.load(args.pre), F.load(args.post)
        responders_only = not args.all_units
        fig_neuron_scatter(pre, post, out, responders_only)
        fig_neuron_distributions(pre, post, out, responders_only)
        fig_diff_summary(pre, post, out, responders_only)
        print_neuron_summary(pre, post, responders_only)
    else:
        print("(--post not given: skipping per-neuron pre/post correlation & difference plots)")

    D = F.load(args.group_data or args.pre)
    if args.data_counts:
        data_counts = load_data_counts(args.data_counts)
        print(f"\nusing REAL data-counts from {args.data_counts}")
    else:
        data_counts = fake_data_counts(args.fake_total, args.fake_seed)
        print(f"\n** no --data-counts given: using a FAKE placeholder dataset "
              f"(n={args.fake_total}, seed={args.fake_seed}) -- replace with real neural "
              f"data via --data-counts group_counts.json when available **")
        print("   fake group counts: " +
              ", ".join(f"{l}={int(c)}" for l, c in zip(GROUP_LABELS, data_counts)))

    fit, best = fig_group_chi2(D, data_counts, out)
    fig_group_proportions_vs_data(D, data_counts, out)
    if fit:
        print("\nchi-squared goodness-of-fit (model responder-group composition vs. data):")
        for m in [x for x in F.MODELS if x in fit]:
            star = "  <- best fit" if m == best else ""
            print(f"  {F.MODELS[m]['label']:32s} chi2={fit[m]['chi2']:7.2f}  p={fit[m]['p']:.4f}{star}")
        print(f"\nbest-fitting representation: {F.MODELS[best]['label']}")


if __name__ == "__main__":
    main()
