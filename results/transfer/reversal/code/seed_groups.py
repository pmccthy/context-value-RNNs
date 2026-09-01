#!/usr/bin/env python
"""Seed-by-seed recovery, and representational metrics compared BETWEEN outcome groups
(recovered vs failed seeds), within each model class.

Recovery metric (recap): `recovered_fraction` = post-reversal mean reward / that same
seed's pre-reversal mean reward (both from history.json). A seed is RECOVERED if
recovered_fraction >= --thr (default 0.8), else FAILED. This is bimodal per seed (near
100% or near ~20%, rarely in between) — see reversal_analysis.py / the README.

This script:
  1. Shows the raw per-seed recovery values (not just the recovered-count summary) —
     the actual bimodal scatter, per class.
  2. Splits each model class's OWN seeds into its recovered vs failed subsets (via
     reversal_analysis.filter_fd) and re-runs the representational-comparison tools
     already built (population_similarity.py's per-neuron pre/post correlation +
     activity-difference, its responder-group composition, and rsa.py's stimulus-only
     RDM) SEPARATELY on each subset, so the two outcome groups can be compared directly.
     A Mann-Whitney U test (per class, where n permits) checks whether the per-neuron
     distributions actually differ between recovered and failed seeds.

  -> seed_recovery_strip            per-seed recovered_fraction, coloured by outcome
  -> seed_group_tuning_correlation  per-neuron pre/post tuning correlation, recovered vs failed
  -> seed_group_activity_diff       per-neuron |post-pre| activity difference, recovered vs failed
  -> seed_group_responder_groups    responder-group composition, recovered vs failed
  -> seed_group_rdm_stim            stimulus-only RDM, one row per outcome group
  -> seed_group_heatmaps            per-unit tuning heatmap, one row per outcome group
  -> seed_group_heatmap_montage_*   every seed's tuning heatmap, tiled per (group, class)
  -> seed_vigour_curves             EVERY seed's vigour-vs-trials trajectory during the
                                    reversal, overlaid, coloured by outcome group — rows
                                    = stimulus, columns = model class

Usage:
  python scripts/16_06_26_seed_groups.py \
         --pre results/figure_data --post results/reversal_2500/figure_data_reversal \
         --reversal-runs results/reversal_2500/model_runs_reversal \
         --out results/reversal_2500/figures_seed_groups
"""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path
import numpy as np, matplotlib; import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import reversal_analysis as RA
    import population_similarity as PS
    import rsa as RSA
except ModuleNotFoundError:
    import importlib
    RA = importlib.import_module("16_06_26_reversal_analysis")
    PS = importlib.import_module("16_06_26_population_similarity")
    RSA = importlib.import_module("16_06_26_rsa")
F = RA.F


# ── recovery table + seed-by-seed view ──────────────────────────────────────────────
def load_recovery_table(post_runs):
    """{model_type: {seed: recovered_fraction}} straight from history.json (no threshold
    applied yet)."""
    table = {}
    for f in glob.glob(str(Path(post_runs) / "*" / "seed*" / "history.json")):
        p = Path(f); mt = p.parent.parent.name; seed = int(p.parent.name[4:])
        rf = json.loads(p.read_text()).get("recovered_fraction")
        if rf is not None:
            table.setdefault(mt, {})[seed] = rf
    return table


def split_groups(table, thr):
    """{model_type: set(seed)} for recovered (>= thr) and failed (< thr)."""
    keep = {mt: {s for s, f in d.items() if f >= thr} for mt, d in table.items()}
    fail = {mt: {s for s, f in d.items() if f < thr} for mt, d in table.items()}
    return keep, fail


def fig_seed_strip(table, thr, out):
    """Per-seed recovered_fraction, one jittered point per seed, coloured/filled by
    outcome (recovered = filled, failed = open), dashed line at the threshold. The
    actual per-seed scatter behind the bimodal-recovery claim, not just its summary."""
    types = [m for m in F.MODELS if m in table]
    if not types:
        print("  (skip seed_recovery_strip: no data)"); return
    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    rng = np.random.default_rng(0)
    for i, mt in enumerate(types):
        seeds = sorted(table[mt])
        fracs = np.array([table[mt][s] for s in seeds]) * 100
        jitter = rng.uniform(-0.16, 0.16, size=len(seeds))
        ok = fracs >= thr * 100
        col = F.MODELS[mt]["color"]
        ax.scatter(i + jitter[ok], fracs[ok], color=col, s=42, edgecolor="white",
                  linewidth=0.6, zorder=3)
        ax.scatter(i + jitter[~ok], fracs[~ok], facecolor="none", edgecolor=col,
                  s=42, linewidth=1.5, zorder=3)
        ax.text(i, -10, f"{int(ok.sum())}/{len(seeds)}\nrecovered", ha="center", fontsize=9.5)
    ax.axhline(thr * 100, color="0.4", lw=1.2, ls="--")
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels([F.MODELS[m]["label"] for m in types], rotation=20, ha="right")
    ax.tick_params(axis="x", labelsize=12)
    ax.set_ylabel("% of pre-reversal reward recovered", fontsize=13); ax.set_ylim(-20, 118)
    legend_elems = [Line2D([0], [0], marker="o", color="w", markerfacecolor="0.35",
                           markeredgecolor="white", label="recovered", markersize=8),
                    Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
                           markeredgecolor="0.35", label="failed", markersize=8)]
    ax.legend(handles=legend_elems, fontsize=10, loc="center left")
    ax.set_title(f"Recovery per seed  ·  dashed = {int(thr*100)}% threshold", fontsize=13)
    fig.tight_layout(); F.save_fig(fig, out, "seed_recovery_strip")


# ── representational metrics, recovered vs failed ───────────────────────────────────
def _group_pair(pre, post, keep, fail, mt, responders_only):
    """(recovered, failed) per-neuron (corr, diff) arrays for one model class, built by
    filtering both pre & post figure_data to each seed subset and reusing
    population_similarity's per-neuron calc."""
    pre_ok, post_ok = RA.filter_fd(pre, keep), RA.filter_fd(post, keep)
    pre_bad, post_bad = RA.filter_fd(pre, fail), RA.filter_fd(post, fail)
    return (PS.per_neuron_pre_post(pre_ok, post_ok, mt, responders_only),
            PS.per_neuron_pre_post(pre_bad, post_bad, mt, responders_only))


def _violin_panel(ax, series, labels, color):
    data = [s[~np.isnan(s)] for s in series if s is not None and np.isfinite(s).any()]
    labs = [l for s, l in zip(series, labels) if s is not None and np.isfinite(s).any()]
    if not data:
        ax.axis("off"); return None
    parts = ax.violinplot(data, showmeans=True, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor(color); pc.set_alpha(0.55)
    parts["cmeans"].set_color("0.15")
    ax.set_xticks(range(1, len(data) + 1)); ax.set_xticklabels(labs, fontsize=10)
    return data


def fig_group_tuning_correlation(pre, post, keep, fail, out, responders_only=True):
    """Per class: violin of the per-neuron pre/post tuning correlation, recovered vs
    failed seeds — does the reversal succeed BECAUSE the tuning reorganises
    differently, or is recovery independent of this representational change? Annotated
    with a Mann-Whitney U p-value where both groups have enough units."""
    types = [m for m in F.MODELS if m in post["model_types"] and m in pre["model_types"]]
    fig, axes = plt.subplots(1, len(types), figsize=(4.2 * len(types), 4.9),
                             sharey=True, squeeze=False)
    for ax, mt in zip(axes[0], types):
        (c_ok, _), (c_bad, _) = _group_pair(pre, post, keep, fail, mt, responders_only)
        n_ok, n_bad = len(keep.get(mt, [])), len(fail.get(mt, []))
        labels = [f"recovered\n({n_ok} seeds)", f"failed\n({n_bad} seeds)"]
        data = _violin_panel(ax, [c_ok, c_bad], labels, F.MODELS[mt]["color"])
        ax.axhline(0, color="0.7", lw=0.8)
        title = F.MODELS[mt]["label"]
        if data and len(data) == 2 and min(len(d) for d in data) >= 5:
            _, p = mannwhitneyu(data[0], data[1])
            title += f"\nMann-Whitney p={p:.1e}"
        ax.set_title(title, fontsize=11)
    axes[0][0].set_ylabel("pre/post tuning correlation", fontsize=13)
    tag = " · pre-reversal responders only" if responders_only else " · all units"
    fig.suptitle("Per-neuron tuning preservation: recovered vs. failed seeds" + tag, fontsize=13)
    fig.tight_layout(); F.save_fig(fig, out, "seed_group_tuning_correlation")


def fig_group_activity_diff(pre, post, keep, fail, out, responders_only=True):
    """Per class: violin of the per-neuron |post - pre| activity difference, recovered
    vs failed seeds."""
    types = [m for m in F.MODELS if m in post["model_types"] and m in pre["model_types"]]
    fig, axes = plt.subplots(1, len(types), figsize=(4.2 * len(types), 4.9),
                             sharey=True, squeeze=False)
    for ax, mt in zip(axes[0], types):
        (_, d_ok), (_, d_bad) = _group_pair(pre, post, keep, fail, mt, responders_only)
        n_ok, n_bad = len(keep.get(mt, [])), len(fail.get(mt, []))
        labels = [f"recovered\n({n_ok} seeds)", f"failed\n({n_bad} seeds)"]
        ad_ok = np.abs(d_ok) if d_ok is not None else None
        ad_bad = np.abs(d_bad) if d_bad is not None else None
        data = _violin_panel(ax, [ad_ok, ad_bad], labels, F.MODELS[mt]["color"])
        title = F.MODELS[mt]["label"]
        if data and len(data) == 2 and min(len(d) for d in data) >= 5:
            _, p = mannwhitneyu(data[0], data[1])
            title += f"\nMann-Whitney p={p:.1e}"
        ax.set_title(title, fontsize=11)
    axes[0][0].set_ylabel("|post − pre| activity", fontsize=13)
    tag = " · pre-reversal responders only" if responders_only else " · all units"
    fig.suptitle("Per-neuron activity-level shift: recovered vs. failed seeds" + tag, fontsize=13)
    fig.tight_layout(); F.save_fig(fig, out, "seed_group_activity_diff")


def fig_group_responder_groups(D, keep, fail, out, stem="seed_group_responder_groups",
                               title_tag="post-reversal"):
    """Per class: responder-group composition (7 mixed-selectivity groups), recovered
    vs failed seeds side by side — solid = recovered, hatched = failed. `D` can be `pre`
    or `post` figure_data (the keep/fail split is always the REVERSAL outcome either
    way); `title_tag`/`stem` should be set together for the pre-reversal call."""
    types = [m for m in F.MODELS if m in D["model_types"]]
    labels = PS.GROUP_LABELS; x = np.arange(len(labels)); w = 0.8 / (2 * len(types))
    fig, ax = plt.subplots(figsize=(13, 5.6))
    for i, mt in enumerate(types):
        d_ok, d_bad = RA.filter_fd(D, keep), RA.filter_fd(D, fail)
        c_ok = PS.group_counts_pooled(d_ok, mt); c_bad = PS.group_counts_pooled(d_bad, mt)
        f_ok = 100 * c_ok / max(c_ok.sum(), 1e-9); f_bad = 100 * c_bad / max(c_bad.sum(), 1e-9)
        off = (i - len(types) / 2 + 0.5) * 2 * w
        ax.bar(x + off - w / 2, f_ok, w, color=F.MODELS[mt]["color"], edgecolor="white",
               label=f"{F.MODELS[mt]['label']} — recovered (n={len(keep.get(mt, []))})")
        ax.bar(x + off + w / 2, f_bad, w, color=F.MODELS[mt]["color"], edgecolor="white",
               hatch="//", alpha=0.75, label=f"{F.MODELS[mt]['label']} — failed (n={len(fail.get(mt, []))})")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.tick_params(axis="x", labelsize=12)
    ax.set_ylabel("% of responsive units")
    ax.set_title(f"Responder-group composition ({title_tag}): recovered vs. failed seeds",
                fontsize=12)
    ax.legend(fontsize=8.5, ncol=2)
    fig.tight_layout(); F.save_fig(fig, out, stem)


BROAD_LABELS = ["pref 0%", "pref 50%", "pref 100%"]


def _broad_counts_pooled(D, mt, seeds):
    """3-category (pref 0%/50%/100%) unit counts, pooled over `seeds`, among units
    RESPONSIVE to at least one stimulus. Uses the same argmax-tuning classification as
    the plain (4-category) alluvial's _unit_categories -- every responsive unit,
    including mixed-selectivity ones, is assigned to its single most-preferred
    stimulus, so this is the "broad" collapse of the 7 fine-grained mixed-selectivity
    groups down to the 3 pure preference categories (silent units excluded, matching
    fig_group_responder_groups' own "% of responsive units" convention)."""
    counts = np.zeros(3)
    if mt not in D["model_types"]:
        return counts
    ti = D["model_types"].index(mt)
    for s in seeds:
        if s not in D["seeds"]:
            continue
        si = D["seeds"].index(s)
        tuning = D["tuning"]["data"][ti, si]
        responsive = D["responsive"]["data"][ti, si]
        if not np.isfinite(tuning).all():
            continue
        cats = _unit_categories(tuning, responsive)   # 0=silent, 1/2/3=pref stim
        for k in range(3):
            counts[k] += int((cats == k + 1).sum())
    return counts


def fig_group_responder_groups_broad(D, keep, fail, out,
                                     stem="seed_group_responder_groups_broad",
                                     title_tag="post-reversal"):
    """Same layout/style as fig_group_responder_groups, but collapsed to the 3 BROAD
    preferred-stimulus categories (0%/50%/100%) instead of the 7 fine-grained mixed-
    selectivity groups -- every responsive unit (including mixed-selectivity ones) is
    assigned to its single argmax-preferred stimulus. Recovered vs failed (solid vs
    hatched), one colour per model class. `D` can be `pre` or `post` figure_data (the
    keep/fail split is always the REVERSAL outcome either way, so a pre-reversal call
    asks "did seeds that go on to recover already look different beforehand?");
    `title_tag`/`stem` should be set together for the pre-reversal call."""
    types = [m for m in F.MODELS if m in D["model_types"]]
    labels = BROAD_LABELS; x = np.arange(len(labels)); w = 0.8 / (2 * len(types))
    if not types:
        print(f"  (skip {stem}: no data)"); return
    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    for i, mt in enumerate(types):
        seeds_ok = sorted(keep.get(mt, set()) & set(D["seeds_present"].get(mt, [])))
        seeds_bad = sorted(fail.get(mt, set()) & set(D["seeds_present"].get(mt, [])))
        c_ok = _broad_counts_pooled(D, mt, seeds_ok)
        c_bad = _broad_counts_pooled(D, mt, seeds_bad)
        f_ok = 100 * c_ok / max(c_ok.sum(), 1e-9); f_bad = 100 * c_bad / max(c_bad.sum(), 1e-9)
        off = (i - len(types) / 2 + 0.5) * 2 * w
        ax.bar(x + off - w / 2, f_ok, w, color=F.MODELS[mt]["color"], edgecolor="white",
               label=f"{F.MODELS[mt]['label']} — recovered (n={len(seeds_ok)})")
        ax.bar(x + off + w / 2, f_bad, w, color=F.MODELS[mt]["color"], edgecolor="white",
               hatch="//", alpha=0.75, label=f"{F.MODELS[mt]['label']} — failed (n={len(seeds_bad)})")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.tick_params(axis="x", labelsize=12)
    ax.set_ylabel("% of responsive units")
    ax.set_title(f"Preferred-stimulus composition (broad, 3 groups, {title_tag}): "
                "recovered vs. failed seeds", fontsize=12)
    ax.legend(fontsize=8.5, ncol=2)
    fig.tight_layout(); F.save_fig(fig, out, stem)


def _strip_panel(ax, x, ys_by_seed, seed_ok_mask, color, rng, jitter=0.16, s=34):
    """One jittered strip at position x: filled = recovered, open = failed (matches
    fig_seed_strip's visual convention throughout the bundle)."""
    ys = np.asarray(ys_by_seed, float); ok = np.asarray(seed_ok_mask, bool)
    j = rng.uniform(-jitter, jitter, size=len(ys))
    if ok.any():
        ax.scatter(x + j[ok], ys[ok], color=color, s=s, edgecolor="white", linewidth=0.5,
                  zorder=3)
    if (~ok).any():
        ax.scatter(x + j[~ok], ys[~ok], facecolor="none", edgecolor=color, s=s,
                  linewidth=1.3, zorder=3)


def fig_group_responder_proportions_seedwise(D, keep, fail, out, stem="seed_responder_proportions",
                                             title_tag="post-reversal"):
    """Per class: proportion of RESPONSIVE units falling in each of the 7 responder
    groups (0%-only, 50%-only, ..., all three), ONE POINT PER SEED (not just the pooled
    bar version in fig_group_responder_groups) -- filled=recovered, open=failed, same
    convention as seed_recovery_strip."""
    types = [m for m in F.MODELS if m in D["model_types"]]
    if not types:
        print(f"  (skip {stem}: no data)"); return
    labels = PS.GROUP_LABELS; x = np.arange(len(labels))
    fig, axes = plt.subplots(1, len(types), figsize=(4.6 * len(types), 5.4), sharey=True,
                             squeeze=False)
    rng = np.random.default_rng(0)
    for ax, mt in zip(axes[0], types):
        seeds = D["seeds_present"].get(mt, [])
        counts = F.responder_group_counts(D, mt)                # (n_seeds_present, 7)
        if counts.size == 0:
            ax.set_title(f"{F.MODELS[mt]['label']}\n(no seeds)", fontsize=11); continue
        frac = 100 * counts / counts.sum(1, keepdims=True).clip(1e-9)  # % of RESPONDERS
        ok_mask = np.array([s in keep.get(mt, set()) for s in seeds])
        for gi in range(len(labels)):
            _strip_panel(ax, gi, frac[:, gi], ok_mask, F.MODELS[mt]["color"], rng)
        n_ok, n_bad = int(ok_mask.sum()), int((~ok_mask).sum())
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_title(f"{F.MODELS[mt]['label']}\n(recovered={n_ok}, failed={n_bad})", fontsize=10.5)
    axes[0][0].set_ylabel("% of responsive units\n(filled = recovered, open = failed)", fontsize=13)
    fig.suptitle(f"Responder-group composition per seed ({title_tag})", fontsize=13)
    fig.tight_layout(); F.save_fig(fig, out, stem)


def fig_group_population_activity_seedwise(D, keep, fail, out, stem="seed_population_activity",
                                           title_tag="post-reversal"):
    """Per class: population activity per stimulus, ONE POINT PER SEED -- filled =
    recovered, open = failed."""
    types = [m for m in F.MODELS if m in D["model_types"]]
    if not types:
        print(f"  (skip {stem}: no data)"); return
    stim = D["stim_labels"]; x = np.arange(len(stim))
    fig, axes = plt.subplots(1, len(types), figsize=(4.0 * len(types), 5.4), sharey=True,
                             squeeze=False)
    rng = np.random.default_rng(0)
    for ax, mt in zip(axes[0], types):
        ti = D["model_types"].index(mt)
        seeds = D["seeds_present"].get(mt, [])
        if not seeds:
            ax.set_title(f"{F.MODELS[mt]['label']}\n(no seeds)", fontsize=11); continue
        seed_idx = [D["seeds"].index(s) for s in seeds]
        act = D["scalars"]["pop_activity"][ti][seed_idx]               # (n_seeds, 3)
        ok_mask = np.array([s in keep.get(mt, set()) for s in seeds])
        for si in range(len(stim)):
            _strip_panel(ax, si, act[:, si], ok_mask, F.MODELS[mt]["color"], rng)
        n_ok, n_bad = int(ok_mask.sum()), int((~ok_mask).sum())
        ax.set_xticks(x); ax.set_xticklabels(stim)
        ax.set_title(f"{F.MODELS[mt]['label']}\n(recovered={n_ok}, failed={n_bad})", fontsize=10.5)
    axes[0][0].set_ylabel("population activity\n(filled = recovered, open = failed)", fontsize=13)
    fig.suptitle(f"Population activity per seed ({title_tag})", fontsize=13)
    fig.tight_layout(); F.save_fig(fig, out, stem)


def fig_group_population_activity_bar(D, keep, fail, out, stem="seed_population_activity_bar",
                                      title_tag="post-reversal"):
    """Per class: population activity per stimulus, mean ± SEM bar (recovered vs failed
    side by side, solid vs hatched) -- bar-chart version of
    fig_group_population_activity_seedwise (same data, same recovered/failed split)."""
    types = [m for m in F.MODELS if m in D["model_types"]]
    if not types:
        print(f"  (skip {stem}: no data)"); return
    stim = D["stim_labels"]; x = np.arange(len(stim)); w = 0.32
    fig, axes = plt.subplots(1, len(types), figsize=(4.0 * len(types), 5.4), sharey=True,
                             squeeze=False)
    for ax, mt in zip(axes[0], types):
        ti = D["model_types"].index(mt)
        seeds = D["seeds_present"].get(mt, [])
        if not seeds:
            ax.set_title(f"{F.MODELS[mt]['label']}\n(no seeds)", fontsize=11); continue
        seed_idx = [D["seeds"].index(s) for s in seeds]
        act = D["scalars"]["pop_activity"][ti][seed_idx]               # (n_seeds, 3)
        ok_mask = np.array([s in keep.get(mt, set()) for s in seeds])
        col = F.MODELS[mt]["color"]
        for mask, off, hatch, alpha in [(ok_mask, -w / 2, None, 1.0), (~ok_mask, w / 2, "//", 0.75)]:
            n = int(mask.sum())
            if n == 0:
                continue
            vals = act[mask]                                          # (n_group, 3)
            m = vals.mean(0)
            sem = vals.std(0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros(vals.shape[1])
            ax.bar(x + off, m, w, yerr=sem, color=col, edgecolor="white", hatch=hatch,
                  alpha=alpha, capsize=3, error_kw=dict(lw=1.2))
        n_ok, n_bad = int(ok_mask.sum()), int((~ok_mask).sum())
        ax.set_xticks(x); ax.set_xticklabels(stim)
        ax.set_title(f"{F.MODELS[mt]['label']}\n(recovered={n_ok}, failed={n_bad})", fontsize=10.5)
    axes[0][0].set_ylabel("population activity\n(mean ± SEM; solid = recovered, hatched = failed)",
                          fontsize=13)
    fig.suptitle(f"Population activity per stimulus ({title_tag})", fontsize=13)
    fig.tight_layout(); F.save_fig(fig, out, stem)


CAT_LABELS = ["silent", "pref 0%", "pref 50%", "pref 100%"]
CAT_COLORS = ["0.65", F.STIM_COLORS[0], F.STIM_COLORS[1], F.STIM_COLORS[2]]


def _unit_categories(tuning, responsive):
    """tuning: (3 stim, H units) stim-window mean activation. responsive: (H, 3) bool
    (paired t-test vs ITI baseline). Returns (H,) int in {0,1,2,3}: 0 = silent (not
    significantly responsive to ANY stimulus), 1/2/3 = preferred stimulus (argmax
    tuning) for units that ARE responsive to at least one -- the same "silent vs
    preferred" split heatmap_montage sorts units by, just made explicit/countable here
    instead of only used for row order."""
    resp_any = responsive.any(1)
    pref = tuning.argmax(0)
    return np.where(resp_any, pref + 1, 0)


def _transition_matrix_4cat(pre, post, mt, seeds):
    """4x4 (pre-category x post-category) unit-COUNT matrix, pooled over `seeds`. Each
    phase's category uses that phase's OWN responsiveness (a unit's "silent" label is
    judged fresh at each timepoint, not carried over)."""
    M = np.zeros((4, 4))
    if mt not in pre["model_types"] or mt not in post["model_types"]:
        return M
    tp, to = pre["model_types"].index(mt), post["model_types"].index(mt)
    for s in seeds:
        if s not in pre["seeds"] or s not in post["seeds"]:
            continue
        pi, qi = pre["seeds"].index(s), post["seeds"].index(s)
        tpre, tpost = pre["tuning"]["data"][tp, pi], post["tuning"]["data"][to, qi]
        if not (np.isfinite(tpre).all() and np.isfinite(tpost).all()):
            continue
        cpre = _unit_categories(tpre, pre["responsive"]["data"][tp, pi])
        cpost = _unit_categories(tpost, post["responsive"]["data"][to, qi])
        for a in range(4):
            for b in range(4):
                M[a, b] += int(((cpre == a) & (cpost == b)).sum())
    return M


def _sankey_ribbon(ax, x0, x1, ytop0, ybot0, ytop1, ybot1, color, alpha=0.5, n=60):
    """A smooth flow band from (x0, [ybot0,ytop0]) to (x1, [ybot1,ytop1])."""
    t = np.linspace(0, 1, n)
    sig = 1 / (1 + np.exp(-12 * (t - 0.5))); sig = (sig - sig[0]) / (sig[-1] - sig[0])
    xs = x0 + (x1 - x0) * t
    top = ytop0 + (ytop1 - ytop0) * sig; bot = ybot0 + (ybot1 - ybot0) * sig
    ax.fill_between(xs, bot, top, color=color, alpha=alpha, lw=0, zorder=1)


def _draw_alluvial(ax, M, labels=CAT_LABELS, colors=CAT_COLORS, gap=0.02, node_w=0.05):
    """M: (k,k) pre-category x post-category unit-count matrix. Two-column flow diagram
    -- left = pre-reversal category composition, right = post -- ribbons (coloured by
    SOURCE/pre category, standard Sankey convention) show how units moved between
    selectivity categories across the reversal."""
    k = M.shape[0]; total = M.sum()
    if total <= 0:
        ax.axis("off"); return
    left_tot, right_tot = M.sum(1), M.sum(0)

    def _spans(totals):
        # bars + gaps must together span exactly [0,1] regardless of k -- otherwise the
        # (k-1)*gap of accumulated whitespace pushes the LAST category below y=0 and off
        # the visible axis (invisible/overlapping-with-next-panel labels for k=8).
        avail = max(1.0 - (k - 1) * gap, 1e-6)
        y, out = 1.0, []
        for f in totals / max(total, 1e-9) * avail:
            out.append((y, y - f)); y = y - f - gap
        return out

    left_spans, right_spans = _spans(left_tot), _spans(right_tot)
    left_cursor = [sp[0] for sp in left_spans]; right_cursor = [sp[0] for sp in right_spans]
    for a in range(k):
        for b in range(k):
            if M[a, b] <= 0:
                continue
            h = M[a, b] / max(total, 1e-9)
            yt0 = left_cursor[a]; yb0 = yt0 - h; left_cursor[a] = yb0
            yt1 = right_cursor[b]; yb1 = yt1 - h; right_cursor[b] = yb1
            _sankey_ribbon(ax, node_w, 1 - node_w, yt0, yb0, yt1, yb1, colors[a])
    for a in range(k):
        y0, y1 = left_spans[a]
        ax.add_patch(Rectangle((0, y1), node_w, max(y0 - y1, 0), color=colors[a],
                               ec="white", lw=0.5))
        if left_tot[a] > 0:
            ax.text(-0.02, (y0 + y1) / 2, f"{labels[a]} ({int(left_tot[a])})", ha="right",
                    va="center", fontsize=8)
    for b in range(k):
        y0, y1 = right_spans[b]
        ax.add_patch(Rectangle((1 - node_w, y1), node_w, max(y0 - y1, 0), color=colors[b],
                               ec="white", lw=0.5))
        if right_tot[b] > 0:
            ax.text(1.02, (y0 + y1) / 2, f"{labels[b]} ({int(right_tot[b])})", ha="left",
                    va="center", fontsize=8)
    ax.set_xlim(-0.62, 1.62); ax.set_ylim(0, 1.02); ax.axis("off")


def fig_group_alluvial(pre, post, keep, fail, out, seed_pref=42, stem="seed_group_alluvial"):
    """2-row (recovered/failed) x model-class grid of a REPRESENTATIVE seed's pre->post
    unit-category alluvial (silent / pref 0% / pref 50% / pref 100%) -- quick-glance
    companion to seed_group_heatmaps."""
    types = [m for m in F.MODELS if m in post["model_types"] and m in pre["model_types"]]
    groups = [("recovered", keep), ("failed", fail)]
    if not types:
        print(f"  (skip {stem}: no data)"); return
    fig, axes = plt.subplots(2, len(types), figsize=(4.6 * len(types), 8.4), squeeze=False)
    any_drawn = False
    for row, (gname, gset) in enumerate(groups):
        for col, mt in enumerate(types):
            ax = axes[row][col]
            seeds = sorted(gset.get(mt, set()) & set(pre["seeds_present"].get(mt, [])) &
                          set(post["seeds_present"].get(mt, [])))
            if not seeds:
                ax.axis("off")
                ax.set_title(f"{F.MODELS[mt]['label']}\n(no {gname} seeds)", fontsize=10)
                continue
            s = seed_pref if seed_pref in seeds else seeds[0]
            _draw_alluvial(ax, _transition_matrix_4cat(pre, post, mt, [s])); any_drawn = True
            ax.set_title(f"{F.MODELS[mt]['label']}\n(seed {s})" if row == 0 else f"seed {s}",
                        fontsize=10)
            if col == 0:
                ax.text(-0.85, 0.5, gname, transform=ax.transAxes, rotation=90,
                        ha="center", va="center", fontsize=13)
    if not any_drawn:
        print(f"  (skip {stem}: no data)"); return
    fig.suptitle("Unit selectivity remapping, pre → post reversal (representative seed)\n"
                 "ribbon coloured by PRE-reversal category", fontsize=12)
    fig.tight_layout(); F.save_fig(fig, out, stem)


def fig_group_alluvial_pooled(pre, post, keep, fail, out, stem="seed_group_alluvial_pooled"):
    """Same layout as fig_group_alluvial, but pooling ALL units across ALL seeds in
    each (outcome group, model class) cell instead of one representative seed -- the
    aggregate remapping pattern, combining neurons across runs."""
    types = [m for m in F.MODELS if m in post["model_types"] and m in pre["model_types"]]
    groups = [("recovered", keep), ("failed", fail)]
    if not types:
        print(f"  (skip {stem}: no data)"); return
    fig, axes = plt.subplots(2, len(types), figsize=(4.6 * len(types), 8.4), squeeze=False)
    any_drawn = False
    for row, (gname, gset) in enumerate(groups):
        for col, mt in enumerate(types):
            ax = axes[row][col]
            seeds = sorted(gset.get(mt, set()) & set(pre["seeds_present"].get(mt, [])) &
                          set(post["seeds_present"].get(mt, [])))
            if not seeds:
                ax.axis("off")
                ax.set_title(f"{F.MODELS[mt]['label']}\n(no {gname} seeds)", fontsize=10)
                continue
            M = _transition_matrix_4cat(pre, post, mt, seeds)
            _draw_alluvial(ax, M); any_drawn = True
            ax.set_title(f"{F.MODELS[mt]['label']}\n({len(seeds)} seeds, {int(M.sum())} units)",
                        fontsize=10)
            if col == 0:
                ax.text(-0.85, 0.5, gname, transform=ax.transAxes, rotation=90,
                        ha="center", va="center", fontsize=13)
    if not any_drawn:
        print(f"  (skip {stem}: no data)"); return
    fig.suptitle("Unit selectivity remapping, pre → post reversal (pooled across seeds)\n"
                 "ribbon coloured by PRE-reversal category", fontsize=12)
    fig.tight_layout(); F.save_fig(fig, out, stem)


def fig_group_alluvial_montage(pre, post, keep, fail, out, ncol=5):
    """Every individual seed's pre->post unit-category alluvial, tiled -- six montages
    (one per outcome-group x model-class combo), mirroring
    fig_group_heatmap_montages/figures.heatmap_montage's convention of showing the
    actual per-seed spread rather than one example or a pooled summary."""
    types = [m for m in F.MODELS if m in post["model_types"] and m in pre["model_types"]]
    for gname, gset in (("recovered", keep), ("failed", fail)):
        for mt in types:
            seeds = sorted(gset.get(mt, set()) & set(pre["seeds_present"].get(mt, [])) &
                          set(post["seeds_present"].get(mt, [])))
            if not seeds:
                print(f"  (skip seed_group_alluvial_montage_{gname}_{mt}: no seeds)"); continue
            nrow = int(np.ceil(len(seeds) / ncol))
            fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.5 * nrow), squeeze=False)
            for i, ax in enumerate(axes.flat):
                if i >= len(seeds):
                    ax.axis("off"); continue
                s = seeds[i]
                _draw_alluvial(ax, _transition_matrix_4cat(pre, post, mt, [s]))
                ax.set_title(f"seed {s}", fontsize=9)
            fig.suptitle(f"{gname.upper()} SEEDS  ·  {F.MODELS[mt]['label']}  ·  "
                        f"per-seed selectivity remapping  [{len(seeds)} seeds]", fontsize=12)
            fig.tight_layout()
            F.save_fig(fig, out, f"seed_group_alluvial_montage_{gname}_{mt}")


# ── mixed-selectivity (all 7 responder groups, not just single-preferred) alluvial ──
def _hex_blend(*hexcolors):
    """Simple RGB average of hex colours (for MIXED-selectivity categories, whose
    natural colour is "between" the stimuli they respond to)."""
    import matplotlib.colors as mcolors
    rgbs = np.array([mcolors.to_rgb(c) for c in hexcolors])
    return tuple(rgbs.mean(0))


# Order matches F.GROUP_ORDER = [(0,),(1,),(2,),(0,1),(0,2),(1,2),(0,1,2)] == PS.GROUP_LABELS.
CAT_LABELS_MIXED = ["silent"] + PS.GROUP_LABELS
CAT_COLORS_MIXED = ["0.65", F.STIM_COLORS[0], F.STIM_COLORS[1], F.STIM_COLORS[2],
                    _hex_blend(F.STIM_COLORS[0], F.STIM_COLORS[1]),
                    _hex_blend(F.STIM_COLORS[0], F.STIM_COLORS[2]),
                    _hex_blend(F.STIM_COLORS[1], F.STIM_COLORS[2]),
                    _hex_blend(F.STIM_COLORS[0], F.STIM_COLORS[1], F.STIM_COLORS[2])]


def _unit_categories_mixed(responsive):
    """responsive: (H, 3) bool (t-test). Returns (H,) int in {0..7}: 0 = silent (not
    responsive to ANY stimulus), 1..7 = index into F.GROUP_ORDER -- i.e. the full
    MIXED-selectivity responder group (0%-only, 50%-only, 100%-only, 0%&50%, 0%&100%,
    50%&100%, all three), not collapsed to a single "preferred" stimulus the way
    _unit_categories is. Same bit-code trick as figures.responder_group_counts."""
    bits = np.array([1, 2, 4])
    codes = responsive.astype(int) @ bits                      # (H,) in 0..7
    code_to_cat = {0: 0}
    for i, c in enumerate(F._GROUP_CODES):                     # [1,2,4,3,5,6,7]
        code_to_cat[c] = i + 1
    return np.array([code_to_cat[c] for c in codes])


def _transition_matrix_mixed(pre, post, mt, seeds):
    """8x8 (pre-category x post-category) unit-count matrix using the FULL
    mixed-selectivity responder-group categorisation (see _unit_categories_mixed),
    pooled over `seeds`."""
    M = np.zeros((8, 8))
    if mt not in pre["model_types"] or mt not in post["model_types"]:
        return M
    tp, to = pre["model_types"].index(mt), post["model_types"].index(mt)
    for s in seeds:
        if s not in pre["seeds"] or s not in post["seeds"]:
            continue
        pi, qi = pre["seeds"].index(s), post["seeds"].index(s)
        cpre = _unit_categories_mixed(pre["responsive"]["data"][tp, pi])
        cpost = _unit_categories_mixed(post["responsive"]["data"][to, qi])
        for a in range(8):
            for b in range(8):
                M[a, b] += int(((cpre == a) & (cpost == b)).sum())
    return M


def fig_group_alluvial_mixed(pre, post, keep, fail, out, seed_pref=42,
                             stem="seed_group_alluvial_mixed"):
    """Same layout as fig_group_alluvial (representative seed, 2x3 grid), but using the
    FULL 7 mixed-selectivity responder groups + silent (8 categories total) instead of
    collapsing multi-stimulus responders to their single argmax-preferred stimulus --
    so a unit that genuinely responds to two stimuli shows up as its own category and
    its own flow, rather than being folded into whichever one it edges out."""
    types = [m for m in F.MODELS if m in post["model_types"] and m in pre["model_types"]]
    groups = [("recovered", keep), ("failed", fail)]
    if not types:
        print(f"  (skip {stem}: no data)"); return
    fig, axes = plt.subplots(2, len(types), figsize=(5.6 * len(types), 9.6), squeeze=False)
    any_drawn = False
    for row, (gname, gset) in enumerate(groups):
        for col, mt in enumerate(types):
            ax = axes[row][col]
            seeds = sorted(gset.get(mt, set()) & set(pre["seeds_present"].get(mt, [])) &
                          set(post["seeds_present"].get(mt, [])))
            if not seeds:
                ax.axis("off")
                ax.set_title(f"{F.MODELS[mt]['label']}\n(no {gname} seeds)", fontsize=10)
                continue
            s = seed_pref if seed_pref in seeds else seeds[0]
            M = _transition_matrix_mixed(pre, post, mt, [s])
            _draw_alluvial(ax, M, labels=CAT_LABELS_MIXED, colors=CAT_COLORS_MIXED,
                          node_w=0.04)
            any_drawn = True
            ax.set_title(f"{F.MODELS[mt]['label']}\n(seed {s})" if row == 0 else f"seed {s}",
                        fontsize=10)
            if col == 0:
                ax.text(-1.05, 0.5, gname, transform=ax.transAxes, rotation=90,
                        ha="center", va="center", fontsize=13)
    if not any_drawn:
        print(f"  (skip {stem}: no data)"); return
    fig.suptitle("Unit selectivity remapping incl. MIXED selectivity, pre → post reversal "
                 "(representative seed)\nribbon coloured by PRE-reversal category", fontsize=12)
    fig.tight_layout(); F.save_fig(fig, out, stem)


def fig_group_alluvial_mixed_pooled(pre, post, keep, fail, out,
                                    stem="seed_group_alluvial_mixed_pooled"):
    """Pooled-across-seeds companion to fig_group_alluvial_mixed (mirrors
    fig_group_alluvial_pooled vs. fig_group_alluvial)."""
    types = [m for m in F.MODELS if m in post["model_types"] and m in pre["model_types"]]
    groups = [("recovered", keep), ("failed", fail)]
    if not types:
        print(f"  (skip {stem}: no data)"); return
    fig, axes = plt.subplots(2, len(types), figsize=(5.6 * len(types), 9.6), squeeze=False)
    any_drawn = False
    for row, (gname, gset) in enumerate(groups):
        for col, mt in enumerate(types):
            ax = axes[row][col]
            seeds = sorted(gset.get(mt, set()) & set(pre["seeds_present"].get(mt, [])) &
                          set(post["seeds_present"].get(mt, [])))
            if not seeds:
                ax.axis("off")
                ax.set_title(f"{F.MODELS[mt]['label']}\n(no {gname} seeds)", fontsize=10)
                continue
            M = _transition_matrix_mixed(pre, post, mt, seeds)
            _draw_alluvial(ax, M, labels=CAT_LABELS_MIXED, colors=CAT_COLORS_MIXED,
                          node_w=0.04)
            any_drawn = True
            ax.set_title(f"{F.MODELS[mt]['label']}\n({len(seeds)} seeds, {int(M.sum())} units)",
                        fontsize=10)
            if col == 0:
                ax.text(-1.05, 0.5, gname, transform=ax.transAxes, rotation=90,
                        ha="center", va="center", fontsize=13)
    if not any_drawn:
        print(f"  (skip {stem}: no data)"); return
    fig.suptitle("Unit selectivity remapping incl. MIXED selectivity, pre → post reversal "
                 "(pooled across seeds)\nribbon coloured by PRE-reversal category", fontsize=12)
    fig.tight_layout(); F.save_fig(fig, out, stem)


def fig_group_rdm_stim(post, keep, fail, out):
    """2-row grid (recovered / failed) x model-class columns of the stimulus-only
    (post-reversal) RDM, so representational geometry can be compared directly between
    outcome groups. Shared colour scale."""
    types = [m for m in F.MODELS if m in post["model_types"]]
    groups = [("recovered", keep), ("failed", fail)]
    rdms = {}
    for gname, gset in groups:
        Dg = RA.filter_fd(post, gset)
        for mt in types:
            rdm, n = RSA._population_rdm_stim(Dg, mt)
            if rdm is not None:
                rdms[(gname, mt)] = (rdm, n)
    if not rdms:
        print("  (skip seed_group_rdm_stim: no data)"); return
    vmax = max(np.nanmax(r) for r, _ in rdms.values())
    stim = post["stim_labels"]
    fig, axes = plt.subplots(2, len(types), figsize=(3.1 * len(types) + 1.0, 6.6), squeeze=False)
    im = None
    for row, (gname, _) in enumerate(groups):
        for col, mt in enumerate(types):
            ax = axes[row][col]
            key = (gname, mt)
            if key not in rdms:
                ax.axis("off"); ax.set_title(f"{F.MODELS[mt]['label']}\n(no {gname} seeds)",
                                             fontsize=10); continue
            rdm, n = rdms[key]
            im = ax.imshow(rdm, cmap="viridis", vmin=0, vmax=vmax, interpolation="nearest")
            ax.set_xticks(range(3)); ax.set_xticklabels(stim, fontsize=9)
            ax.set_yticks(range(3))
            ax.set_yticklabels(stim if col == 0 else [], fontsize=9)
            for a in range(3):
                for b in range(3):
                    ax.text(b, a, f"{rdm[a, b]:.2f}", ha="center", va="center", fontsize=9,
                            color="w" if rdm[a, b] < vmax * 0.6 else "k")
            if row == 0:
                ax.set_title(F.MODELS[mt]["label"], fontsize=11)
            if col == 0:
                ax.set_ylabel(f"{gname}\n(n={n} units)", fontsize=10)
    fig.colorbar(im, ax=axes, shrink=0.7, pad=0.02, label="dissimilarity (1 − r)")
    fig.suptitle("Post-reversal stimulus-only RDM: recovered vs. failed seeds", fontsize=13)
    F.save_fig(fig, out, "seed_group_rdm_stim")


def fig_group_heatmaps(post, keep, fail, out, seed_pref=42):
    """2-row grid (recovered / failed) x model-class columns of a per-unit,
    stim-window tuning ACTIVATION heatmap (unit x stimulus, units sorted by preferred
    stimulus) — one representative seed per cell, so you can see actual individual
    seeds' representations, not just their pooled summary. Each cell uses `seed_pref`
    if that seed is in the (model, group)'s set, else its smallest available seed.
    Shared colour scale across all panels."""
    types = [m for m in F.MODELS if m in post["model_types"]]
    groups = [("recovered", keep), ("failed", fail)]
    panels, all_vals = {}, []
    for gname, gset in groups:
        for mt in types:
            seeds = sorted(gset.get(mt, set()) & set(post["seeds_present"].get(mt, [])))
            if not seeds:
                continue
            s = seed_pref if seed_pref in seeds else seeds[0]
            ti, si = post["model_types"].index(mt), post["seeds"].index(s)
            md = post["tuning"]["data"][ti, si]                     # (3 stim, H units)
            if not np.isfinite(md).all():
                continue
            panels[(gname, mt)] = (md, s)
            all_vals.append(md)
    if not panels:
        print("  (skip seed_group_heatmaps: no data)"); return
    vmax = float(np.nanpercentile(np.concatenate([m.ravel() for m in all_vals]), 99))
    stim = post["stim_labels"]
    fig, axes = plt.subplots(2, len(types), figsize=(3.2 * len(types) + 1.0, 7.4), squeeze=False)
    im = None
    for row, (gname, _) in enumerate(groups):
        for col, mt in enumerate(types):
            ax = axes[row][col]
            key = (gname, mt)
            if key not in panels:
                ax.axis("off")
                ax.set_title(f"{F.MODELS[mt]['label']}\n(no {gname} seeds)", fontsize=10)
                continue
            md, s = panels[key]
            peak = md.max(0); pref = md.argmax(0); silent = peak < 0.05 * peak.max()
            order = np.lexsort((-peak, np.where(silent, 3, pref)))
            im = ax.imshow(md[:, order].T, aspect="auto", cmap="hot", vmin=0, vmax=vmax,
                           interpolation="nearest")
            ax.set_xticks(range(3)); ax.set_xticklabels(stim, fontsize=9)
            ax.set_title(f"{F.MODELS[mt]['label']}\n(seed {s})" if row == 0 else f"seed {s}",
                        fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{gname}\nhidden unit (sorted)", fontsize=10)
            else:
                ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.7, pad=0.02, label="mean activation")
    fig.suptitle("Post-reversal per-unit tuning heatmaps — recovered vs. failed seeds\n"
                 "(one representative seed per cell)", fontsize=12)
    F.save_fig(fig, out, "seed_group_heatmaps")


def fig_group_heatmap_montages(post, keep, fail, out, ncol=6):
    """Per (outcome group, model class): the FULL per-seed tuning-heatmap montage
    (figures.py's heatmap_montage) restricted to that group's seeds — every recovered
    seed's heatmap tiled together, and separately every failed seed's, per class, so you
    can see the actual within-group seed-to-seed spread rather than one example. Shared
    colour scale across all 6 montages. Saved as
    seed_group_heatmap_montage_<recovered|failed>_<model_type>."""
    types = [m for m in F.MODELS if m in post["model_types"]]
    vmax = F.tuning_vmax(post, types)
    for gname, gset in (("recovered", keep), ("failed", fail)):
        Dg = RA.filter_fd(post, gset)
        for mt in types:
            n = len(gset.get(mt, set()))
            title = (f"{gname.upper()} SEEDS  ·  {F.MODELS[mt]['label']}  ·  "
                     f"per-unit stim-window activation (sorted by preferred stim)  "
                     f"[{n} seeds]")
            fig = F.heatmap_montage(Dg, mt, vmax=vmax, ncol=ncol, title=title)
            if fig is None:
                print(f"  (skip seed_group_heatmap_montage_{gname}_{mt}: no seeds)")
                continue
            F.save_fig(fig, out, f"seed_group_heatmap_montage_{gname}_{mt}")


def _load_phase_seed(runs, key, include_onset=False):
    """{model_type: {'n_trials': int, 'seeds': [(seed, trials_x, vals (n_probe,3))]}}
    for one training phase (original OR reversal), PER-SEED (not averaged) — the
    per-seed analogue of reversal_analysis.py's _load_phase, same trials-seen x-axis
    convention (probe_update * n_trials / total_updates) so pre- and post-reversal
    segments join up on one continuous axis.

    include_onset=True (only meaningful for a REVERSAL-phase `runs`) additionally
    prepends a trials=0 point from a `history_onset.json` sidecar next to history.json,
    if present -- the deterministic probe evaluated on the model EXACTLY as it started
    the reversal, before any reversal-phase gradient update (see
    reversal_onset_probe.py). Without this, the first plotted post-reversal point is
    already ~--probe-every updates (tens of trials) into reversal training, which can
    look like the curve "starts" already partway through its transient."""
    acc = {}
    for f in glob.glob(str(Path(runs) / "*" / "seed*" / "history.json")):
        p = Path(f); mt = p.parent.parent.name; seed = int(p.parent.name[4:])
        h = json.loads(p.read_text())
        if not h.get("probe_update") or not h.get("probe_" + key):
            continue          # e.g. probe_rpe absent from pre-RPE-instrumentation runs
        tot = max(h.get("total_updates", 1), 1); ntr = h.get("n_trials", 2500)
        trials = np.asarray(h["probe_update"], float) * ntr / tot
        vals = np.asarray(h["probe_" + key], float)
        if include_onset:
            onset_f = p.parent / "history_onset.json"
            if onset_f.exists():
                onset = json.loads(onset_f.read_text()).get(key)
                if onset is not None:
                    trials = np.concatenate([[0.0], trials])
                    vals = np.concatenate([np.asarray(onset, float)[None, :], vals], axis=0)
        d = acc.setdefault(mt, {"n_trials": ntr, "seeds": []})
        d["seeds"].append((seed, trials, vals))
    return acc


def fig_seed_curves(pre_runs, post_runs, keep, fail, out, key="vigour", ylabel="mean vigour",
                    title="Lick vigour", stem="seed_vigour_curves",
                    phase_note="original training → reversal (dashed = reversal)"):
    """EVERY seed's <key> trajectory across the WHOLE experiment — original training
    then the reversal, on ONE continuous trials axis with the reversal onset marked —
    the same convention as reversal_analysis.py's fig_combined_timeline (used for
    reversal_timeline_vigour etc): each stimulus is its usual STIM_COLORS colour, legend
    "stimulus (pre -> post)" via REV_LABELS, one line per seed drawn individually rather
    than averaged into a mean +/- SEM band. The outcome-group comparison is by ROW
    (recovered / failed) instead of by colour, since colour is already stimulus here —
    columns = model class."""
    PRE = _load_phase_seed(pre_runs, key)
    POST = _load_phase_seed(post_runs, key, include_onset=True)
    types = [m for m in RA.F.MODELS if m in PRE and m in POST]
    if not types:
        print(f"  (skip {stem}: need probed original AND reversal runs)"); return
    groups = [("recovered", keep), ("failed", fail)]
    fig, axes = plt.subplots(2, len(types), figsize=(4.9 * len(types), 8.4),
                             sharex="col", sharey=True, squeeze=False)
    for col, mt in enumerate(types):
        n_pre = PRE[mt]["n_trials"]
        post_by_seed = {s: (t, v) for s, t, v in POST[mt]["seeds"]}
        for row, (gname, gset) in enumerate(groups):
            ax = axes[row][col]
            n = 0
            for seed, tpre, vpre in PRE[mt]["seeds"]:
                if seed not in post_by_seed or seed not in gset.get(mt, set()):
                    continue
                tpost, vpost = post_by_seed[seed]
                n += 1
                for s in range(3):
                    c = F.STIM_COLORS[s]
                    ax.plot(tpre, vpre[:, s], color=c, lw=1.0, alpha=0.45)
                    ax.plot(tpost + n_pre, vpost[:, s], color=c, lw=1.0, alpha=0.45)
            ax.axvline(n_pre, color="0.3", lw=1.5, ls="--", zorder=5)
            # x tight (as everywhere else), but a little y headroom — otherwise seeds
            # saturating exactly at vigour=1.0/0.0 hug the panel edge and can read as a
            # top/bottom border even though the spine itself is off (axes.spines.top:
            # False in the style sheet — this is real data sitting on the boundary, not
            # a border artefact).
            ax.margins(x=0, y=0.04)
            if col == 0:
                ax.set_ylabel(f"{gname}\n{ylabel}")               # default axes.labelsize
            # n= is PER (row, col) cell -- it differs by model class (e.g. RL only can
            # have 0 failed seeds while another class has several) -- so it has to be a
            # per-panel title, not folded into the shared (sharey) row ylabel, which
            # would only ever show column 0's count for the whole row.
            if row == 0:
                ax.set_title(f"{F.MODELS[mt]['label']}\n(n={n})", fontsize=11)
            else:
                ax.set_title(f"(n={n})", fontsize=11)
            if row == 1:
                ax.set_xlabel("trials")                          # default axes.labelsize
    legend_elems = [Line2D([0], [0], color=F.STIM_COLORS[s], lw=2.2, label=RA.REV_LABELS[s])
                    for s in range(3)] + \
                   [Line2D([0], [0], color="0.3", lw=1.5, ls="--", label="reversal onset")]
    axes[0][-1].legend(handles=legend_elems, fontsize=9.5, title="stimulus (pre → post)",
                       title_fontsize=9.5, loc="best")
    fig.suptitle(f"{title}, every seed  ·  {phase_note}\n"
                 f"rows = outcome group", fontsize=12)
    fig.tight_layout(); F.save_fig(fig, out, stem)


def fig_seed_curves_avg(pre_runs, post_runs, keep, fail, out, key="vigour",
                        ylabel="mean vigour", title="Lick vigour",
                        stem="seed_vigour_curves_avg",
                        phase_note="original training → reversal (dashed = reversal)"):
    """Group MEAN +/- SEM version of fig_seed_curves: identical rows=outcome-group /
    cols=model-class / STIM_COLORS layout, but averaging over the seeds within each
    group (band = SEM) instead of drawing every seed's line individually -- the
    non-seed-resolved companion to fig_seed_curves, for when the individual-seed
    spaghetti is too dense to read a group-level trend off of."""
    PRE = _load_phase_seed(pre_runs, key)
    POST = _load_phase_seed(post_runs, key, include_onset=True)
    types = [m for m in RA.F.MODELS if m in PRE and m in POST]
    if not types:
        print(f"  (skip {stem}: need probed original AND reversal runs)"); return
    groups = [("recovered", keep), ("failed", fail)]
    fig, axes = plt.subplots(2, len(types), figsize=(4.9 * len(types), 8.4),
                             sharex="col", sharey=True, squeeze=False)
    for col, mt in enumerate(types):
        n_pre = PRE[mt]["n_trials"]
        post_by_seed = {s: (t, v) for s, t, v in POST[mt]["seeds"]}
        for row, (gname, gset) in enumerate(groups):
            ax = axes[row][col]
            pre_list, post_list = [], []
            for seed, tpre, vpre in PRE[mt]["seeds"]:
                if seed not in post_by_seed or seed not in gset.get(mt, set()):
                    continue
                pre_list.append((tpre, vpre)); post_list.append(post_by_seed[seed])
            n = len(pre_list)
            if n:
                Lp = min(v.shape[0] for _, v in pre_list)
                xp = pre_list[0][0][:Lp]
                Vp = np.stack([v[:Lp] for _, v in pre_list])          # (n, Lp, 3)
                Lq = min(v.shape[0] for _, v in post_list)
                xq = post_list[0][0][:Lq] + n_pre
                Vq = np.stack([v[:Lq] for _, v in post_list])
                for s in range(3):
                    c = F.STIM_COLORS[s]
                    for x, V in ((xp, Vp), (xq, Vq)):
                        m = V[:, :, s].mean(0); sem = V[:, :, s].std(0) / np.sqrt(n)
                        ax.fill_between(x, m - sem, m + sem, color=c, alpha=0.22, lw=0)
                        ax.plot(x, m, color=c, lw=2.0)
            ax.axvline(n_pre, color="0.3", lw=1.5, ls="--", zorder=5)
            ax.margins(x=0, y=0.06)
            if col == 0:
                ax.set_ylabel(f"{gname}\n{ylabel}")
            # n= is per (row, col) cell -- see fig_seed_curves for why it can't live in
            # the shared row ylabel.
            if row == 0:
                ax.set_title(f"{F.MODELS[mt]['label']}\n(n={n})", fontsize=11)
            else:
                ax.set_title(f"(n={n})", fontsize=11)
            if row == 1:
                ax.set_xlabel("trials")
    legend_elems = [Line2D([0], [0], color=F.STIM_COLORS[s], lw=2.2, label=RA.REV_LABELS[s])
                    for s in range(3)] + \
                   [Line2D([0], [0], color="0.3", lw=1.5, ls="--", label="reversal onset")]
    axes[0][-1].legend(handles=legend_elems, fontsize=9.5, title="stimulus (pre → post)",
                       title_fontsize=9.5, loc="best")
    fig.suptitle(f"{title}, group mean ± SEM  ·  {phase_note}\n"
                 f"rows = outcome group", fontsize=12)
    fig.tight_layout(); F.save_fig(fig, out, stem)


def fig_condition_comparison_curves(pre_runs, panels, out, key="vigour", ylabel="mean vigour",
                                    title="Lick vigour", stem="condition_comparison_vigour"):
    """Compare a SINGLE stimulus's <key>-vs-trials trajectory across reward_scale_by_stim
    conditions (control / boost_up / damp_down, see run_reward_scale_intervention.sh),
    group mean +/- SEM across seeds. The pre-reversal segment is IDENTICAL across
    conditions (every condition warm-starts from the same original models), so it's
    drawn ONCE per panel in a neutral colour; each condition's post-reversal segment
    then forks off in its own colour right at the reversal line -- makes the causal
    effect of the intervention visually direct rather than needing to eyeball three
    separately-generated figures side by side.

    panels: list of (panel_title, stim_index, [(cond_label, cond_runs_dir, color), ...])
    -- typically one panel per intervention direction, e.g. [("0%->100% (boost_up)", 0,
    [("control", ...), ("boost_up 2x", ...)]), ("100%->0% (damp_down)", 2, [("control",
    ...), ("damp_down 0.3x", ...)])]. Rows = model class, columns = one per panel."""
    types = list(F.MODELS)
    ncols = len(panels)
    fig, axes = plt.subplots(len(types), ncols, figsize=(4.8 * ncols, 3.3 * len(types)),
                             sharex="col", squeeze=False)
    PRE = _load_phase_seed(pre_runs, key)
    for col, (ptitle, stim, conditions) in enumerate(panels):
        for row, mt in enumerate(types):
            ax = axes[row][col]
            if mt not in PRE:
                ax.axis("off"); continue
            n_pre = PRE[mt]["n_trials"]
            pre_list = [(tpre, vpre) for _, tpre, vpre in PRE[mt]["seeds"]]
            if pre_list:
                Lp = min(v.shape[0] for _, v in pre_list)
                xp = pre_list[0][0][:Lp]
                Vp = np.stack([v[:Lp, stim] for _, v in pre_list])
                m = Vp.mean(0); sem = Vp.std(0) / np.sqrt(len(pre_list))
                ax.fill_between(xp, m - sem, m + sem, color="0.55", alpha=0.25, lw=0)
                ax.plot(xp, m, color="0.35", lw=1.8, label="pre-reversal (shared)")
            for clabel, cdir, ccolor in conditions:
                POST = _load_phase_seed(cdir, key, include_onset=True)
                if mt not in POST:
                    continue
                post_list = [(t, v) for _, t, v in POST[mt]["seeds"]]
                if not post_list:
                    continue
                Lq = min(v.shape[0] for _, v in post_list)
                xq = post_list[0][0][:Lq] + n_pre
                Vq = np.stack([v[:Lq, stim] for _, v in post_list])
                n = Vq.shape[0]
                m = Vq.mean(0); sem = Vq.std(0) / np.sqrt(n)
                ax.fill_between(xq, m - sem, m + sem, color=ccolor, alpha=0.22, lw=0)
                ax.plot(xq, m, color=ccolor, lw=2.2, label=f"{clabel} (n={n})")
            ax.axvline(n_pre, color="0.3", lw=1.3, ls="--", zorder=5)
            ax.margins(x=0, y=0.06)
            if row == 0:
                ax.set_title(ptitle, fontsize=11)
            if col == 0:
                ax.set_ylabel(f"{F.MODELS[mt]['label']}\n{ylabel}", fontsize=10)
            if row == len(types) - 1:
                ax.set_xlabel("trials", fontsize=11)
            ax.legend(fontsize=7.5, loc="best")
    fig.suptitle(f"{title}: reward-scale intervention vs. control  ·  group mean ± SEM",
                fontsize=13)
    fig.tight_layout(); F.save_fig(fig, out, stem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", default="results/figure_data")
    ap.add_argument("--post", default="results/reversal_5000/figure_data_reversal")
    ap.add_argument("--reversal-runs", default="results/reversal_5000/model_runs_reversal")
    ap.add_argument("--pre-runs", default="results/model_runs",
                    help="original-training run dirs WITH probes, for the combined "
                         "pre->post seed-curve figure")
    ap.add_argument("--thr", type=float, default=0.8)
    ap.add_argument("--all-units", action="store_true")
    ap.add_argument("--out", default="figures_seed_groups")
    ap.add_argument("--style", default=str(F.DEFAULT_STYLE))
    args = ap.parse_args()
    matplotlib.use("Agg")
    if Path(args.style).exists():
        plt.style.use(args.style)
    else:
        print(f"** WARNING: style file not found at {args.style} — figures will use "
              f"default matplotlib styling, not the house style **")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    table = load_recovery_table(args.reversal_runs)
    keep, fail = split_groups(table, args.thr)
    print(f"seed groups @ thr={args.thr:.0%}:")
    for mt in [m for m in F.MODELS if m in table]:
        print(f"  {F.MODELS[mt]['label']:32s} recovered={len(keep.get(mt, []))}  "
              f"failed={len(fail.get(mt, []))}")
    fig_seed_strip(table, args.thr, out)

    pre, post = F.load(args.pre), F.load(args.post)
    responders_only = not args.all_units
    fig_group_tuning_correlation(pre, post, keep, fail, out, responders_only)
    fig_group_activity_diff(pre, post, keep, fail, out, responders_only)
    fig_group_responder_groups(post, keep, fail, out)
    fig_group_responder_groups(pre, keep, fail, out, stem="seed_group_responder_groups_pre",
                               title_tag="pre-reversal")
    fig_group_responder_groups_broad(post, keep, fail, out)
    fig_group_responder_groups_broad(pre, keep, fail, out,
                                     stem="seed_group_responder_groups_broad_pre",
                                     title_tag="pre-reversal")
    fig_group_responder_proportions_seedwise(post, keep, fail, out)
    fig_group_population_activity_seedwise(post, keep, fail, out)
    fig_group_population_activity_bar(post, keep, fail, out)
    fig_group_rdm_stim(post, keep, fail, out)
    fig_group_heatmaps(post, keep, fail, out)
    fig_group_heatmap_montages(post, keep, fail, out)
    fig_group_alluvial(pre, post, keep, fail, out)
    fig_group_alluvial_pooled(pre, post, keep, fail, out)
    fig_group_alluvial_montage(pre, post, keep, fail, out)
    fig_group_alluvial_mixed(pre, post, keep, fail, out)
    fig_group_alluvial_mixed_pooled(pre, post, keep, fail, out)
    fig_seed_curves(args.pre_runs, args.reversal_runs, keep, fail, out,
                    key="vigour", ylabel="mean vigour", title="Lick vigour")
    fig_seed_curves(args.pre_runs, args.reversal_runs, keep, fail, out,
                    key="rpe", ylabel="mean RPE (TD error)",
                    title="Reward-prediction error", stem="seed_rpe_curves")
    fig_seed_curves_avg(args.pre_runs, args.reversal_runs, keep, fail, out,
                        key="vigour", ylabel="mean vigour", title="Lick vigour")
    fig_seed_curves_avg(args.pre_runs, args.reversal_runs, keep, fail, out,
                        key="rpe", ylabel="mean RPE (TD error)",
                        title="Reward-prediction error", stem="seed_rpe_curves_avg")
    fig_seed_curves(args.pre_runs, args.reversal_runs, keep, fail, out,
                    key="value", ylabel="critic value estimate V(s)",
                    title="Critic value prediction", stem="seed_value_curves")
    fig_seed_curves_avg(args.pre_runs, args.reversal_runs, keep, fail, out,
                        key="value", ylabel="critic value estimate V(s)",
                        title="Critic value prediction", stem="seed_value_curves_avg")


if __name__ == "__main__":
    main()
