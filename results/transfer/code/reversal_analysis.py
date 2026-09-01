#!/usr/bin/env python
"""Reversal experiment analysis: performance recovery + representation remapping.

Reads the pre-reversal figure_data (--pre), the post-reversal figure_data (--post) and
the reversal run dirs (--reversal-runs, for the recovery learning curves). Saves:

  reversal_recovery            mean reward on the reversed task vs update, per class,
                               with each class's pre-reversal performance (dotted) — does
                               it climb back to pre-reversal performance?
  reversal_pref_transition     for each class, a 3x3 matrix of how units' PREFERRED
                               stimulus (by input) maps pre -> post reversal (over units
                               that were responders pre-reversal). A DIAGONAL means the
                               representation kept its stimulus-identity tuning (value
                               re-read on top); an ANTI-DIAGONAL (0<->100%) means the code
                               itself followed the value swap.

Usage:
  python scripts/16_06_26_reversal_analysis.py \
         --pre results/16_06_26_final/figure_data_v2 \
         --post results/16_06_26_final/figure_data_reversal \
         --reversal-runs results/16_06_26_final/model_runs_reversal \
         --out results/16_06_26_final/figures_reversal
"""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path
import numpy as np, matplotlib; import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import figures as F
except ModuleNotFoundError:
    import importlib
    F = importlib.import_module("16_06_26_figures")

# stimulus labels showing the reversal swap (pre value -> post value); 50% is unchanged
REV_LABELS = ["0% → 100%", "50% → 50%", "100% → 0%"]

_KEEP = None   # {model_type: set(seeds)} to restrict analysis to recovered seeds; None = all


def _ok(mt, seed):
    return _KEEP is None or seed in _KEEP.get(mt, set())


def compute_recovered(post_runs, thr):
    """Which seeds recovered >= thr of their pre-reversal reward. Returns
    (keep {model_type: set(seeds)}, total {model_type: n_seeds}) using each reversal run's
    history.json 'recovered_fraction' (= post-reversal reward / pre-reversal reward)."""
    keep, total = {}, {}
    for f in glob.glob(str(Path(post_runs) / "*" / "seed*" / "history.json")):
        p = Path(f); mt = p.parent.parent.name; seed = int(p.parent.name[4:])
        rf = json.loads(p.read_text()).get("recovered_fraction")
        total[mt] = total.get(mt, 0) + 1
        if rf is not None and rf >= thr:
            keep.setdefault(mt, set()).add(seed)
    return keep, total


def filter_fd(D, keep):
    """Copy of a figure_data dict restricted to `keep` seeds per model_type: seeds_present
    is trimmed and excluded (type, seed) slices are NaN'd (responsive -> False) so the
    figure helpers ignore them."""
    ti = {m: i for i, m in enumerate(D["model_types"])}
    si = {s: i for i, s in enumerate(D["seeds"])}
    mask = np.zeros((len(D["model_types"]), len(D["seeds"])), bool)
    for m in D["model_types"]:
        for s in D["seeds_present"].get(m, []):
            if s not in keep.get(m, set()):
                mask[ti[m], si[s]] = True
    Dg = dict(D)
    Dg["seeds_present"] = {m: [s for s in D["seeds_present"].get(m, []) if s in keep.get(m, set())]
                           for m in D["model_types"]}
    S = dict(D["scalars"])
    for k, v in D["scalars"].items():
        if isinstance(v, np.ndarray):
            vv = v.copy(); vv[mask] = np.nan; S[k] = vv
    Dg["scalars"] = S
    for blk, fill in (("tuning", np.nan), ("aligned_mean", np.nan), ("responsive", False)):
        b = dict(D[blk]); arr = D[blk]["data"].copy(); arr[mask] = fill; b["data"] = arr; Dg[blk] = b
    return Dg


def fig_recovery_proportion(keep, total, thr, out):
    """Bar per class: % of seeds that recovered >= thr of pre-reversal reward (count on bar)."""
    types = [m for m in F.MODELS if m in total]
    x = np.arange(len(types))
    frac = [100.0 * len(keep.get(m, set())) / max(total[m], 1) for m in types]
    fig, ax = plt.subplots(figsize=(6.8, 5))
    ax.bar(x, frac, 0.6, color=[F.MODELS[m]["color"] for m in types], edgecolor="white")
    for xi, m, fr in zip(x, types, frac):
        ax.text(xi, fr + 2, f"{len(keep.get(m, set()))}/{total[m]}", ha="center", fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels([F.MODELS[m]["label"] for m in types], rotation=20, ha="right")
    ax.tick_params(axis="x", labelsize=12)
    ax.set_ylabel("% of seeds recovered"); ax.set_ylim(0, 108)
    ax.set_title(f"Seeds recovering ≥ {int(round(thr * 100))}% of pre-reversal reward", fontsize=13)
    fig.tight_layout(); F.save_fig(fig, out, "reversal_recovered_proportion")


def load_histories(reversal_runs):
    """{model_type: {'curves': [(updates, rewards)], 'pre': [..]}} from history.json files."""
    H = {}
    for f in glob.glob(str(Path(reversal_runs) / "*" / "seed*" / "history.json")):
        p = Path(f); mt = p.parent.parent.name
        if not _ok(mt, int(p.parent.name[4:])):
            continue
        h = json.loads(p.read_text())
        d = H.setdefault(mt, {"curves": [], "pre": []})
        d["curves"].append((np.asarray(h["update"], float), np.asarray(h["mean_reward"], float)))
        d["pre"].append(h["pre_reversal_reward"])
    return H


def _smooth(y, w=15):
    """Rolling mean (valid) for readable learning curves."""
    return np.convolve(y, np.ones(w) / w, mode="valid") if len(y) >= w else y


def fig_reversal_recovery(reversal_runs, out, smooth=15):
    """Recovery learning curves (reversed-task reward vs update), per class, with the
    pre-reversal performance as a dotted reference line."""
    H = load_histories(reversal_runs)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for mt in [m for m in F.MODELS if m in H]:
        d = H[mt]; col = F.MODELS[mt]["color"]
        L = min(len(u) for u, _ in d["curves"])
        R = np.stack([r[:L] for _, r in d["curves"]]); U = d["curves"][0][0][:L]
        m = _smooth(R.mean(0), smooth); s = _smooth(R.std(0) / np.sqrt(len(R)), smooth)
        Us = U[len(U) - len(m):]
        ax.plot(Us, m, color=col, lw=2.2, label=f"{F.MODELS[mt]['label']} (n={len(R)})")
        ax.fill_between(Us, m - s, m + s, color=col, alpha=0.2, lw=0)
        ax.axhline(np.mean(d["pre"]), color=col, lw=1.2, ls=":")   # pre-reversal performance
    ax.set_xlabel("reversal update"); ax.set_ylabel("mean reward (reversed task)")
    ax.set_title("Reversal recovery  ·  dotted = pre-reversal performance", fontsize=15)
    ax.legend(fontsize=12); ax.margins(x=0)
    fig.tight_layout(); F.save_fig(fig, out, "reversal_recovery")


def fig_recovery_fraction(reversal_runs, out, smooth=15):
    """Recovery expressed as % of each seed's OWN pre-reversal reward (normalised per
    seed, then averaged over seeds ± SEM) — 100% = fully recovered. Makes the class
    comparison fair despite slightly different per-class baselines."""
    H = {}
    for f in glob.glob(str(Path(reversal_runs) / "*" / "seed*" / "history.json")):
        p = Path(f); mt = p.parent.parent.name; h = json.loads(p.read_text())
        if not _ok(mt, int(p.parent.name[4:])):
            continue
        pre = h.get("pre_reversal_reward"); mr = h.get("mean_reward")
        if not mr or not pre or pre <= 1e-9:
            continue
        H.setdefault(mt, []).append((np.asarray(h["update"], float),
                                     np.asarray(mr, float) / pre * 100.0))
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for mt in [m for m in F.MODELS if m in H]:
        curves = H[mt]; L = min(len(u) for u, _ in curves)
        R = np.stack([c[:L] for _, c in curves]); U = curves[0][0][:L]
        m = _smooth(R.mean(0), smooth); s = _smooth(R.std(0) / np.sqrt(len(R)), smooth)
        Us = U[len(U) - len(m):]
        ax.plot(Us, m, color=F.MODELS[mt]["color"], lw=2.2,
                label=f"{F.MODELS[mt]['label']} (n={len(R)})")
        ax.fill_between(Us, m - s, m + s, color=F.MODELS[mt]["color"], alpha=0.2, lw=0)
    ax.axhline(100, color="0.5", lw=1.2, ls=":")                 # fully recovered
    ax.set_xlabel("reversal update"); ax.set_ylabel("% of pre-reversal reward")
    ax.set_title("Reversal recovery (% of pre-reversal performance)", fontsize=14)
    ax.legend(fontsize=12); ax.margins(x=0)
    fig.tight_layout(); F.save_fig(fig, out, "reversal_recovery_fraction")


def fig_pref_transition(pre, post, out):
    """Per class: 3x3 pre->post preferred-stimulus transition (row-normalised), over units
    that were responders pre-reversal."""
    stim = pre["stim_labels"]
    types = [m for m in F.MODELS if m in post["model_types"] and m in pre["model_types"]]
    fig, axes = plt.subplots(1, len(types), figsize=(3.9 * len(types) + 1.2, 4.6),
                             squeeze=False)
    im = None
    for k, (ax, mt) in enumerate(zip(axes[0], types)):
        tp, to = pre["model_types"].index(mt), post["model_types"].index(mt)
        seeds = sorted(set(pre["seeds_present"][mt]) & set(post["seeds_present"][mt]))
        M = np.zeros((3, 3))
        for s in seeds:
            tpre = pre["tuning"]["data"][tp, pre["seeds"].index(s)]      # (stim, unit)
            tpost = post["tuning"]["data"][to, post["seeds"].index(s)]
            resp = pre["responsive"]["data"][tp, pre["seeds"].index(s)].any(1)  # responder pre
            pr, po = tpre.argmax(0), tpost.argmax(0)                     # preferred stim pre/post
            for a in range(3):
                for b in range(3):
                    M[a, b] += np.sum((pr == a) & (po == b) & resp)
        M = M / M.sum(1, keepdims=True).clip(1e-9)                      # P(post pref | pre pref)
        im = ax.imshow(M, cmap="magma", vmin=0, vmax=1)
        ax.set_xticks(range(3)); ax.set_xticklabels(stim)
        ax.set_yticks(range(3)); ax.set_yticklabels(stim if k == 0 else [])
        ax.tick_params(labelsize=13)
        ax.set_title(f"{F.MODELS[mt]['label']}\n({len(seeds)} seeds)", fontsize=12)
        for a in range(3):
            for b in range(3):
                ax.text(b, a, f"{M[a, b]:.2f}", ha="center", va="center", fontsize=11,
                        color="w" if M[a, b] < 0.6 else "k")
    fig.supxlabel("post-reversal preferred", fontsize=15)
    fig.supylabel("pre-reversal preferred", fontsize=15)
    fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02, label="fraction")
    fig.suptitle("Preferred-stimulus remapping (pre → post reversal)\n"
                 "diagonal = identity kept · anti-diagonal = code followed the value swap",
                 fontsize=13)
    F.save_fig(fig, out, "reversal_pref_transition")


def _matched_seeds(pre, post, mt):
    return sorted(set(pre["seeds_present"][mt]) & set(post["seeds_present"][mt]))


def fig_tuning_correlation(pre, post, out):
    """Per class: mean per-unit correlation between a unit's pre- and post-reversal tuning
    vector (over the 3 stimuli), across pre-reversal responders. 1 = tuning unchanged
    (identity kept), low/negative = tuning rotated (code followed the value swap)."""
    types = [m for m in F.MODELS if m in post["model_types"] and m in pre["model_types"]]
    x = np.arange(len(types)); means, sems, cols = [], [], []
    for mt in types:
        tp, to = pre["model_types"].index(mt), post["model_types"].index(mt)
        per_seed = []
        for s in _matched_seeds(pre, post, mt):
            a = pre["tuning"]["data"][tp, pre["seeds"].index(s)]       # (3, H)
            b = post["tuning"]["data"][to, post["seeds"].index(s)]
            resp = pre["responsive"]["data"][tp, pre["seeds"].index(s)].any(1)
            a, b = a[:, resp], b[:, resp]                              # (3, n)
            ac, bc = a - a.mean(0), b - b.mean(0)
            denom = np.sqrt((ac ** 2).sum(0) * (bc ** 2).sum(0)).clip(1e-9)
            per_seed.append(np.nanmean((ac * bc).sum(0) / denom))
        means.append(np.mean(per_seed)); sems.append(np.std(per_seed) / np.sqrt(max(len(per_seed), 1)))
        cols.append(F.MODELS[mt]["color"])
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.bar(x, means, 0.6, yerr=sems, capsize=4, color=cols, edgecolor="white")
    ax.axhline(0, color="0.6", lw=1)
    ax.set_xticks(x); ax.set_xticklabels([F.MODELS[m]["label"] for m in types], rotation=20, ha="right")
    ax.tick_params(axis="x", labelsize=12)
    ax.set_ylabel("pre/post tuning correlation"); ax.set_ylim(-1.05, 1.05)
    ax.set_title("Tuning preserved across reversal  (1 = unchanged, <0 = flipped)", fontsize=13)
    fig.tight_layout(); F.save_fig(fig, out, "reversal_tuning_correlation")


def fig_pre_post_bars(pre, post, key, ylabel, title, stem, out):
    """Per class: pre (grey) vs post (colour) bars of a metric per stimulus, matched seeds."""
    types = [m for m in F.MODELS if m in post["model_types"] and m in pre["model_types"]]
    stim = pre["stim_labels"]; x = np.arange(3); w = 0.38
    fig, axes = plt.subplots(1, len(types), figsize=(4.3 * len(types), 4.4),
                             sharey=True, squeeze=False)
    for ax, mt in zip(axes[0], types):
        tp, to = pre["model_types"].index(mt), post["model_types"].index(mt)
        seeds = _matched_seeds(pre, post, mt)
        A = pre["scalars"][key][tp][[pre["seeds"].index(s) for s in seeds]]     # (n,3)
        B = post["scalars"][key][to][[post["seeds"].index(s) for s in seeds]]
        n = max(len(seeds), 1)
        ax.bar(x - w / 2, A.mean(0), w, yerr=A.std(0) / np.sqrt(n), capsize=3,
               color="0.72", edgecolor="white", label="pre")
        ax.bar(x + w / 2, B.mean(0), w, yerr=B.std(0) / np.sqrt(n), capsize=3,
               color=F.MODELS[mt]["color"], edgecolor="white", label="post")
        ax.set_xticks(x); ax.set_xticklabels(stim)
        ax.set_title(f"{F.MODELS[mt]['label']}\n(n={len(seeds)})", fontsize=11)
    axes[0][0].set_ylabel(ylabel); axes[0][-1].legend(fontsize=11)
    fig.suptitle(title, fontsize=13); fig.tight_layout()
    F.save_fig(fig, out, stem)


def _load_probes(reversal_runs, key):
    """{model_type: (updates, array (n_seed, n_probe, 3))} for one probe metric."""
    acc = {}
    for f in glob.glob(str(Path(reversal_runs) / "*" / "seed*" / "history.json")):
        p = Path(f); mt = p.parent.parent.name; h = json.loads(p.read_text())
        if not h.get("probe_update") or not _ok(mt, int(p.parent.name[4:])):
            continue
        d = acc.setdefault(mt, {"u": np.asarray(h["probe_update"]), "vals": []})
        d["vals"].append(np.asarray(h["probe_" + key], float))
    res = {}
    for mt, d in acc.items():
        L = min(v.shape[0] for v in d["vals"])
        res[mt] = (d["u"][:L], np.stack([v[:L] for v in d["vals"]]))
    return res


def fig_metric_timeline(pre, reversal_runs, out, key, pre_key, ylabel, title, stem):
    """Per class: each stimulus's metric vs reversal update (mean over seeds); the
    pre-reversal level is a dotted horizontal per stimulus and the reversal onset is the
    dashed line at update 0."""
    P = _load_probes(reversal_runs, key)
    types = [m for m in F.MODELS if m in P]
    tp = {m: i for i, m in enumerate(pre["model_types"])}
    fig, axes = plt.subplots(1, len(types), figsize=(4.7 * len(types), 4.5),
                             sharey=True, squeeze=False)
    for ax, mt in zip(axes[0], types):
        u, arr = P[mt]; m = arr.mean(0); sem = arr.std(0) / np.sqrt(arr.shape[0])  # (n_probe, 3)
        pre_val = np.nanmean(pre["scalars"][pre_key][tp[mt]], 0)      # (3,)
        for s in range(3):
            c = F.STIM_COLORS[s]
            ax.fill_between(u, m[:, s] - sem[:, s], m[:, s] + sem[:, s], color=c, alpha=0.18, lw=0)
            ax.plot(u, m[:, s], color=c, lw=2.2, label=REV_LABELS[s])
            ax.axhline(pre_val[s], color=c, lw=1, ls=":")
        ax.axvline(0, color="0.4", lw=1.3, ls="--")
        ax.set_xlim(-0.03 * u.max(), u.max()); ax.set_xlabel("reversal update")
        ax.set_title(f"{F.MODELS[mt]['label']} (n={arr.shape[0]})", fontsize=11)
    axes[0][0].set_ylabel(ylabel)
    axes[0][-1].legend(title="stimulus (pre → post)", fontsize=9.5, title_fontsize=9.5)
    fig.suptitle(f"{title}  ·  dotted = pre-reversal level, dashed = reversal (band = SEM)",
                 fontsize=12)
    fig.tight_layout(); F.save_fig(fig, out, stem)


def fig_pre_post_heatmaps(pre, post, out, seed=None):
    """Per class: pre (top) vs post (bottom) per-unit tuning heatmap for one seed, units
    sorted by their PRE-reversal preferred stimulus (same order both rows, so the
    reorganisation is visible). Shared colour scale. `seed` defaults to 42 if available
    (recovered), else the first seed present in every shown class."""
    types = [m for m in F.MODELS if m in post["model_types"] and m in pre["model_types"]]
    common = None
    for m in types:
        s = set(pre["seeds_present"].get(m, [])) & set(post["seeds_present"].get(m, []))
        common = s if common is None else (common & s)
    common = common or set()
    if seed is None or seed not in common:
        seed = 42 if 42 in common else (min(common) if common else None)
    if seed is None:
        print("  (skip heatmaps: no common seed)"); return
    mats = []
    for mt in types:
        mats += [pre["tuning"]["data"][pre["model_types"].index(mt), pre["seeds"].index(seed)],
                 post["tuning"]["data"][post["model_types"].index(mt), post["seeds"].index(seed)]]
    vmax = float(np.nanpercentile(np.concatenate([m.ravel() for m in mats]), 99))
    fig, axes = plt.subplots(2, len(types), figsize=(3.0 * len(types) + 1.0, 6.4), squeeze=False)
    im = None
    for j, mt in enumerate(types):
        A = pre["tuning"]["data"][pre["model_types"].index(mt), pre["seeds"].index(seed)]
        B = post["tuning"]["data"][post["model_types"].index(mt), post["seeds"].index(seed)]
        peak = A.max(0); pref = A.argmax(0); silent = peak < 0.05 * peak.max()
        order = np.lexsort((-peak, np.where(silent, 3, pref)))
        for r, (M, lab) in enumerate([(A, "pre"), (B, "post")]):
            ax = axes[r][j]
            im = ax.imshow(M[:, order].T, aspect="auto", cmap="hot", vmin=0, vmax=vmax,
                           interpolation="nearest")
            ax.set_xticks(range(3)); ax.set_xticklabels(pre["stim_labels"], fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{lab}-reversal\nhidden unit (pre-sorted)", fontsize=11)
            else:
                ax.set_yticks([])
            if r == 0:
                ax.set_title(F.MODELS[mt]["label"], fontsize=11)
    fig.colorbar(im, ax=axes, shrink=0.6, label="mean activation")
    fig.suptitle(f"Per-unit tuning pre vs post reversal (seed {seed}, units sorted by "
                 f"pre-reversal preferred)", fontsize=12)
    F.save_fig(fig, out, "reversal_heatmaps_prepost")


def _load_phase(runs, key):
    """{model_type: {'n_trials': int, 'seeds': [(trials_x, vals (n_probe,3))]}} for one
    training phase (original OR reversal), converting probe-update -> trials seen."""
    acc = {}
    for f in glob.glob(str(Path(runs) / "*" / "seed*" / "history.json")):
        p = Path(f); mt = p.parent.parent.name; h = json.loads(p.read_text())
        if not h.get("probe_update") or not _ok(mt, int(p.parent.name[4:])):
            continue
        tot = max(h.get("total_updates", 1), 1); ntr = h.get("n_trials", 2500)
        trials = np.asarray(h["probe_update"], float) * ntr / tot
        d = acc.setdefault(mt, {"n_trials": ntr, "seeds": []})
        d["seeds"].append((trials, np.asarray(h["probe_" + key], float)))
    return acc


def fig_combined_timeline(pre_runs, post_runs, out, key, ylabel, title, stem):
    """Per class: a metric per stimulus over the WHOLE experiment on one trials axis —
    original training then the reversal, joined, with the reversal point dashed."""
    PRE, POST = _load_phase(pre_runs, key), _load_phase(post_runs, key)
    types = [m for m in F.MODELS if m in PRE and m in POST]
    if not types:
        print(f"  (skip {stem}: need probed original AND reversal runs)"); return
    fig, axes = plt.subplots(1, len(types), figsize=(4.9 * len(types), 4.6),
                             sharey=True, squeeze=False)

    def stats(phase, mt, x_off):
        """mean ± SEM over seeds; x shifted by x_off (trials)."""
        L = min(v.shape[0] for _, v in phase[mt]["seeds"])
        x = phase[mt]["seeds"][0][0][:L] + x_off
        V = np.stack([vv[:L] for _, vv in phase[mt]["seeds"]])             # (n_seed, L, 3)
        n = V.shape[0]
        return x, V.mean(0), V.std(0) / np.sqrt(n), n

    for ax, mt in zip(axes[0], types):
        n_pre = PRE[mt]["n_trials"]
        xp, mp, sp, n = stats(PRE, mt, 0.0); xq, mq, sq, _ = stats(POST, mt, n_pre)
        for s in range(3):
            c = F.STIM_COLORS[s]
            for x, m, sem in ((xp, mp, sp), (xq, mq, sq)):
                ax.fill_between(x, m[:, s] - sem[:, s], m[:, s] + sem[:, s], color=c, alpha=0.18, lw=0)
            ax.plot(xp, mp[:, s], color=c, lw=2.2)
            ax.plot(xq, mq[:, s], color=c, lw=2.2, label=REV_LABELS[s])    # legend on post seg
        ax.axvline(n_pre, color="0.3", lw=1.5, ls="--")
        ax.margins(x=0); ax.set_xlabel("trials")
        ax.set_title(f"{F.MODELS[mt]['label']} (n={n})", fontsize=11)
    axes[0][0].set_ylabel(ylabel)
    axes[0][-1].legend(title="stimulus (pre → post)", fontsize=9.5, title_fontsize=9.5)
    fig.suptitle(f"{title}  ·  original training → reversal (dashed = reversal; band = SEM)",
                 fontsize=12)
    fig.tight_layout(); F.save_fig(fig, out, stem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", default="figure_data")
    ap.add_argument("--post", default="figure_data_reversal")
    ap.add_argument("--reversal-runs", default="model_runs_reversal")
    ap.add_argument("--pre-runs", default=None,
                    help="original-training run dirs WITH probes (for the combined pre->post "
                         "timeline); default = --reversal-runs's sibling 'model_runs'")
    ap.add_argument("--post-runs", default=None, help="reversal run dirs with probes "
                    "(default = --reversal-runs)")
    ap.add_argument("--out", default="figures_reversal")
    ap.add_argument("--min-recovery", type=float, default=0.0,
                    help="if >0, restrict EVERY figure to seeds whose recovered_fraction >= this "
                         "(e.g. 0.8) and report the proportion recovered per class")
    ap.add_argument("--style", default=str(F.DEFAULT_STYLE))
    args = ap.parse_args()
    matplotlib.use("Agg")
    if Path(args.style).exists():
        plt.style.use(args.style)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    pre, post = F.load(args.pre), F.load(args.post)
    pre_runs = args.pre_runs or "model_runs"
    post_runs = args.post_runs or args.reversal_runs

    global _KEEP
    if args.min_recovery > 0:                      # restrict to seeds that recovered
        keep, total = compute_recovered(post_runs, args.min_recovery)
        _KEEP = keep
        print(f"recovered >= {args.min_recovery:.0%} of pre-reversal reward:")
        for m in [x for x in F.MODELS if x in total]:
            print(f"  {m:24s} {len(keep.get(m, set()))}/{total[m]} seeds")
        fig_recovery_proportion(keep, total, args.min_recovery, out)
        pre, post = filter_fd(pre, keep), filter_fd(post, keep)

    fig_reversal_recovery(args.reversal_runs, out)
    fig_recovery_fraction(args.reversal_runs, out)
    fig_pref_transition(pre, post, out)
    fig_tuning_correlation(pre, post, out)
    # pre-vs-post end-of-training bars
    fig_pre_post_bars(pre, post, "vigour", "mean vigour", "Lick vigour: pre vs post reversal",
                      "reversal_bars_vigour", out)
    fig_pre_post_bars(pre, post, "pop_activity", "mean activation",
                      "Population activity: pre vs post reversal", "reversal_bars_activity", out)
    fig_pre_post_bars(pre, post, "n_responsive", "# responsive units",
                      "Responder counts: pre vs post reversal", "reversal_bars_responders", out)
    # metric time courses. If the ORIGINAL runs were probed we draw the full
    # original-training -> reversal timeline on one trials axis (reversal marked);
    # otherwise we fall back to the reversal phase only (pre level as a dotted reference).
    combined = bool(_load_phase(pre_runs, "vigour"))
    for key, yl, title, stem in [
            ("vigour", "mean vigour", "Lick vigour", "reversal_timeline_vigour"),
            ("pop_activity", "mean activation", "Population activity", "reversal_timeline_activity"),
            ("frac_responsive", "fraction responsive", "Selectivity", "reversal_timeline_selectivity")]:
        if combined:
            fig_combined_timeline(pre_runs, post_runs, out, key, yl, title, stem)
        else:
            fig_metric_timeline(pre, post_runs, out, key, key, yl, title + " over reversal", stem)
    fig_pre_post_heatmaps(pre, post, out)


if __name__ == "__main__":
    main()
