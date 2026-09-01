#!/usr/bin/env python
"""Time-to-recovery: how many trials into the reversal does performance climb back to
(>= --thr of) its pre-reversal level, on a SMOOTHED curve (to avoid a lucky single
noisy update looking like "recovery"). Two independent versions of "recovery":

  reward  -- the usual definition (recovered_fraction, see reversal_analysis.py /
             seed_groups.py), but resolved IN TIME rather than as one pre/post
             endpoint ratio. Uses the FINE-GRAINED per-update mean_reward history
             (every optimizer update, not just every probe_every) logged by
             train_reversal.py, smoothed with a rolling mean over --reward-smooth
             updates.
  vigour  -- a vigour-only analog, since reward and vigour need not recover in lockstep
             (that's exactly what this compares). Physical stimulus identity (0/1/2) is
             unchanged by the reversal -- only which one is high/low value flips -- so
             define the "reversed-value span" S_t = vigour[stim 0] - vigour[stim 2] at
             each post-reversal probe (this is POSITIVE once the agent has correctly
             re-learned the flip, since originally-low-value stim 0 is now the
             high-value one). Recovery = S_t reaching --thr of the seed's own converged
             PRE-reversal span S_pre = vigour_pre[stim 2] - vigour_pre[stim 0] (the
             analogous, un-reversed quantity). Uses the coarser probe_vigour history
             (every probe_every updates), smoothed over --vigour-smooth probes.

Both need the run to have been trained with --probe-every > 0 (for vigour) /
train_reversal.py's per-update history (always present) for reward.

  -> recovery_time    trials-to-recovery distributions, reward-based vs vigour-based,
                       one column per model class (jittered strip + median tick)

Usage:
  python code/recovery_time.py --pre-runs results/model_runs \
         --post-runs results/reversal_5000/model_runs_reversal \
         --out results/reversal_5000/figures_seed_groups
"""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path
import numpy as np, matplotlib; import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import seed_groups as SG
except ModuleNotFoundError:
    import importlib
    SG = importlib.import_module("16_06_26_seed_groups")
F = SG.RA.F


def _smooth(y, window):
    """Centered moving average with an edge correction (divide by actual in-window
    overlap, not the nominal kernel size) so the ends aren't pulled toward zero."""
    y = np.asarray(y, float)
    if window <= 1 or len(y) < 2:
        return y
    k = min(int(window), len(y))
    kernel = np.ones(k) / k
    num = np.convolve(y, kernel, mode="same")
    denom = np.convolve(np.ones_like(y), kernel, mode="same")
    return num / denom


def time_to_recovery_reward(post_runs, thr=0.8, smooth_window=51):
    """{model_type: {seed: trials_to_recovery}}, NaN if the smoothed curve never
    crosses thr * pre_reversal_reward."""
    out = {}
    for f in glob.glob(str(Path(post_runs) / "*" / "seed*" / "history.json")):
        p = Path(f); mt = p.parent.parent.name; seed = int(p.parent.name[4:])
        h = json.loads(p.read_text())
        upd, rew, pre_r = h.get("update"), h.get("mean_reward"), h.get("pre_reversal_reward")
        if not upd or not rew or not pre_r or pre_r <= 1e-9:
            continue
        tot = max(h.get("total_updates", len(upd)), 1); ntr = h.get("n_trials", 2500)
        trials = np.asarray(upd, float) * ntr / tot
        sm = _smooth(rew, smooth_window)
        cross = np.where(sm >= thr * pre_r)[0]
        ttr = float(trials[cross[0]]) if len(cross) else float("nan")
        out.setdefault(mt, {})[seed] = ttr
    return out


def time_to_recovery_vigour(pre_runs, post_runs, thr=0.8, smooth_window=3):
    """{model_type: {seed: trials_to_recovery}}, NaN if the smoothed reversed-value
    vigour span never reaches thr * the seed's own converged pre-reversal span."""
    PRE = SG._load_phase_seed(pre_runs, "vigour")
    POST = SG._load_phase_seed(post_runs, "vigour")
    out = {}
    for mt in PRE:
        if mt not in POST:
            continue
        post_by_seed = {s: (t, v) for s, t, v in POST[mt]["seeds"]}
        for seed, tpre, vpre in PRE[mt]["seeds"]:
            if seed not in post_by_seed:
                continue
            tpost, vpost = post_by_seed[seed]
            s_pre = float(vpre[-1, 2] - vpre[-1, 0])       # converged pre-reversal span
            if s_pre <= 1e-6:
                continue                                    # never learned the task at all
            s_t = vpost[:, 0] - vpost[:, 2]                 # reversed-value span over time
            sm = _smooth(s_t, smooth_window)
            cross = np.where(sm / s_pre >= thr)[0]
            ttr = float(tpost[cross[0]]) if len(cross) else float("nan")
            out.setdefault(mt, {})[seed] = ttr
    return out


def time_to_recovery_vigour_target(pre_runs, post_runs, thr=0.2, smooth_window=3):
    """{model_type: {seed: trials_to_recovery}}, based on how closely post-reversal
    vigour matches the MIRROR-SWAPPED pre-reversal target level, for ALL THREE stimuli
    (not just the reversed-value span between stim 0 and stim 2). At each post-reversal
    probe t:

        target = [v_pre[-1, 2], v_pre[-1, 1], v_pre[-1, 0]]     (swap stim 0 <-> stim 2,
                                                                   since only their VALUE
                                                                   is swapped by the
                                                                   reversal, stim 1 is
                                                                   untouched)
        err_t  = mean(|vpost[t] - target|)                       (mean abs error, 3 stim)

    err_t is normalized by the seed's own converged PRE-reversal span
    S_pre = v_pre[-1, 2] - v_pre[-1, 0], NOT by the raw target values themselves. The
    raw targets are near zero for the devalued stimulus post-reversal (originally-100%
    stim 2, now 0%) -- dividing by a near-zero target would blow tiny absolute noise up
    into huge relative error exactly where recovery matters most to detect, i.e. the
    numerical-instability concern you flagged. Normalizing by the fixed, well-conditioned
    pre-reversal span avoids that. Recovered once the smoothed err_t / |S_pre| first
    drops to <= thr (default: within 20% of the seed's own pre-reversal dynamic range,
    averaged across all 3 stimuli)."""
    PRE = SG._load_phase_seed(pre_runs, "vigour")
    POST = SG._load_phase_seed(post_runs, "vigour")
    out = {}
    for mt in PRE:
        if mt not in POST:
            continue
        post_by_seed = {s: (t, v) for s, t, v in POST[mt]["seeds"]}
        for seed, tpre, vpre in PRE[mt]["seeds"]:
            if seed not in post_by_seed:
                continue
            tpost, vpost = post_by_seed[seed]
            v_pre_final = vpre[-1]                          # (3,) converged pre-reversal vigour
            s_pre = float(v_pre_final[2] - v_pre_final[0])
            if abs(s_pre) <= 1e-6:
                continue                                    # never learned the task at all
            target = v_pre_final[[2, 1, 0]]                 # mirror swap: 0<->2, 1 unchanged
            err_t = np.abs(vpost - target[None, :]).mean(1) # (n_probes,) mean |err| over 3 stim
            sm = _smooth(err_t, smooth_window)
            cross = np.where(sm / abs(s_pre) <= thr)[0]
            ttr = float(tpost[cross[0]]) if len(cross) else float("nan")
            out.setdefault(mt, {})[seed] = ttr
    return out


def time_to_recovery_vigour_target_per_stim(pre_runs, post_runs, thr=0.2, smooth_window=3,
                                            stim=0):
    """Single-stimulus version of time_to_recovery_vigour_target: {model_type: {seed:
    trials_to_recovery}}, using ONLY `stim`'s own error against its mirror-swapped target
    (no averaging across all 3 stimuli), so the 0%->100% direction (stim 0) and the
    100%->0% direction (stim 2) can be timed SEPARATELY instead of jointly. This is what
    makes a per-stimulus comparison possible at all -- the reward-based TTR
    (time_to_recovery_reward) is a single scalar per seed (reward is summed over all
    stimuli in a trial block), so it cannot be split by which stimulus flipped value.

    Same normalization as the combined version: err_t (now just |vpost[:, stim] -
    target|, not mean(1)) is divided by the seed's own converged pre-reversal reversed-
    value span |S_pre| = |v_pre[-1,2] - v_pre[-1,0]|, so both stimuli's recovery times are
    on the same, well-conditioned scale despite one target being near-1 and the other
    near-0."""
    mirror = {0: 2, 1: 1, 2: 0}
    PRE = SG._load_phase_seed(pre_runs, "vigour")
    POST = SG._load_phase_seed(post_runs, "vigour")
    out = {}
    for mt in PRE:
        if mt not in POST:
            continue
        post_by_seed = {s: (t, v) for s, t, v in POST[mt]["seeds"]}
        for seed, tpre, vpre in PRE[mt]["seeds"]:
            if seed not in post_by_seed:
                continue
            tpost, vpost = post_by_seed[seed]
            v_pre_final = vpre[-1]
            s_pre = float(v_pre_final[2] - v_pre_final[0])
            if abs(s_pre) <= 1e-6:
                continue                                    # never learned the task at all
            target = v_pre_final[mirror[stim]]              # scalar target for THIS stim
            err_t = np.abs(vpost[:, stim] - target)         # (n_probes,), no mean(1)
            sm = _smooth(err_t, smooth_window)
            cross = np.where(sm / abs(s_pre) <= thr)[0]
            ttr = float(tpost[cross[0]]) if len(cross) else float("nan")
            out.setdefault(mt, {})[seed] = ttr
    return out


def fig_recovery_time(panels, out, fname="recovery_time"):
    """panels: list of (title, ttr_dict, thr_text) tuples, one column per panel. thr_text
    is a pre-formatted string describing what "reached" means for that definition (e.g.
    "80%" for a >= crossing, "within 20%" for a <= error crossing) -- kept as free text
    since the two families of definition cross in opposite directions."""
    types = [m for m in F.MODELS if any(m in ttr for _, ttr, _ in panels)]
    if not types:
        print(f"  (skip {fname}: no data)"); return
    panel_w = (2.6 * len(types) + 1.5) / 2       # matches the original 2-panel per-panel width
    fig, axes = plt.subplots(1, len(panels), figsize=(panel_w * len(panels), 5.4),
                             squeeze=False)
    axes = axes[0]
    rng = np.random.default_rng(0)
    for ax, (title, ttr, thr_text) in zip(axes, panels):
        xticklabels = []
        for i, mt in enumerate(types):
            d = ttr.get(mt, {})
            vals = np.array([v for v in d.values() if np.isfinite(v)])
            col = F.MODELS[mt]["color"]
            if len(vals):
                jitter = rng.uniform(-0.16, 0.16, size=len(vals))
                ax.scatter(i + jitter, vals, color=col, s=36, edgecolor="white",
                          linewidth=0.5, alpha=0.85, zorder=3)
                ax.scatter([i], [np.median(vals)], color="black", marker="_", s=280,
                          linewidth=2.2, zorder=4)
            xticklabels.append(f"{F.MODELS[mt]['label']}\n({len(vals)}/{len(d)} reached "
                               f"{thr_text})")
        ax.set_xticks(range(len(types)))
        ax.set_xticklabels(xticklabels, rotation=20, ha="right", fontsize=9)
        ax.set_title(title, fontsize=12)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("trials to recovery")
    fig.suptitle("Time to recovery  ·  first crossing on a smoothed curve (dash = median)",
                 fontsize=12)
    fig.tight_layout(); F.save_fig(fig, out, fname)


def fig_recovery_definition_comparison(reward_table, thr, vigour_ttr, target_ttr, out,
                                       target_thr, fname="recovery_definition_comparison"):
    """Grouped bar: % of seeds classified RECOVERED under each of three independent
    definitions of "recovery" -- (1) canonical reward-based recovered_fraction >= thr;
    (2) vigour reversed-value-span reaching thr of its pre-reversal span; (3) vigour
    target-matching, all 3 stimuli close to their mirror-swapped pre-reversal targets --
    per model class. Also prints a per-class reclassification contingency against the
    canonical reward-based definition (which seeds flip pass/fail depending on which
    definition you use)."""
    types = [m for m in F.MODELS if m in reward_table]
    if not types:
        print(f"  (skip {fname}: no data)"); return
    defs = [
        ("reward\n(recovered_fraction)",
         {mt: {s: f >= thr for s, f in reward_table.get(mt, {}).items()} for mt in types}),
        ("vigour\n(reversed-span)",
         {mt: {s: np.isfinite(v) for s, v in vigour_ttr.get(mt, {}).items()} for mt in types}),
        ("vigour\n(target-match)",
         {mt: {s: np.isfinite(v) for s, v in target_ttr.get(mt, {}).items()} for mt in types}),
    ]
    fig, ax = plt.subplots(figsize=(2.4 * len(types) + 2, 5.2))
    w = 0.26; x = np.arange(len(types))
    colors = ["0.35", "0.6", F.MODELS[types[0]]["color"] if types else "C2"]
    colors = ["0.25", "0.55", "0.8"]
    for j, (label, d) in enumerate(defs):
        pcts, ns = [], []
        for mt in types:
            vals = list(d.get(mt, {}).values())
            pcts.append(100 * np.mean(vals) if vals else np.nan)
            ns.append(len(vals))
        bars = ax.bar(x + (j - 1) * w, pcts, width=w * 0.9, label=label, color=colors[j],
                      edgecolor="white", linewidth=0.6)
        for xi, p, n in zip(x + (j - 1) * w, pcts, ns):
            if np.isfinite(p):
                ax.text(xi, p + 2, f"n={n}", ha="center", fontsize=7.5, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([F.MODELS[m]["label"] for m in types], rotation=20, ha="right",
                       fontsize=11)
    ax.set_ylabel("% of seeds classified recovered", fontsize=13)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_ylim(0, 122)
    ax.legend(fontsize=9, loc="upper right", frameon=False)
    ax.set_title(f"Recovery rate by definition  ·  reward thr={int(thr*100)}%, "
                f"vigour-span thr={int(thr*100)}%, target-match thr=within "
                f"{int(target_thr*100)}%", fontsize=11)
    fig.tight_layout(); F.save_fig(fig, out, fname)

    print("\n  Reclassification vs canonical reward-based definition (n seeds where both "
          "definitions have a verdict):")
    for label, d in defs[1:]:
        print(f"  -- {label.replace(chr(10), ' ')} --")
        for mt in types:
            rew, alt = defs[0][1].get(mt, {}), d.get(mt, {})
            common = sorted(set(rew) & set(alt))
            if not common:
                continue
            agree_rec = sum(rew[s] and alt[s] for s in common)
            agree_fail = sum((not rew[s]) and (not alt[s]) for s in common)
            rew_only = sum(rew[s] and not alt[s] for s in common)
            alt_only = sum((not rew[s]) and alt[s] for s in common)
            excl = len(set(rew) | set(alt)) - len(common)
            excl_txt = f"  ({excl} excluded: no valid pre-reversal span)" if excl else ""
            print(f"    {F.MODELS[mt]['label']:32s} n={len(common):2d}  "
                  f"agree-recovered={agree_rec:2d}  agree-failed={agree_fail:2d}  "
                  f"reward-pass-only={rew_only:2d}  alt-pass-only={alt_only:2d}{excl_txt}")


def rpe_vs_recovery_time(post_runs, thr=0.8, reward_smooth=51, window_trials=500, stim=0):
    """{model_type: [(seed, ttr, mean_abs_rpe_early)]}, one row per seed that actually
    reached reward-based recovery (finite TTR from time_to_recovery_reward -- "time to
    recovery" is only defined for seeds that got there at all). mean_abs_rpe_early =
    mean |probe_rpe[stim]| over the first `window_trials` of the reversal, for `stim`
    (default 0 = the stimulus whose value newly flips low->high) -- does the SIZE of the
    early reward-prediction-error signal predict how FAST the seed goes on to recover?

    Needs genuine live probe_rpe (cxval.vigour.infer_rpe wired into make_probe) --
    NOT available for runs trained before that wiring was added (e.g. reversal_5000),
    which only have the terminal/frozen-model RPE workaround (a repeated-evaluation
    noise cloud around a fixed point, not a real early-vs-late trajectory, so unusable
    for this specific question)."""
    ttr_table = time_to_recovery_reward(post_runs, thr=thr, smooth_window=reward_smooth)
    out = {}
    for f in glob.glob(str(Path(post_runs) / "*" / "seed*" / "history.json")):
        p = Path(f); mt = p.parent.parent.name; seed = int(p.parent.name[4:])
        if mt not in ttr_table or seed not in ttr_table[mt]:
            continue
        ttr = ttr_table[mt][seed]
        if not np.isfinite(ttr):
            continue
        h = json.loads(p.read_text())
        pu, rpe = h.get("probe_update"), h.get("probe_rpe")
        if not pu or not rpe:
            continue
        tot = max(h.get("total_updates", len(pu)), 1); ntr = h.get("n_trials", 2500)
        trials = np.asarray(pu, float) * ntr / tot
        rpe = np.asarray(rpe, float)                    # (n_probes, 3)
        mask = trials <= window_trials
        if not mask.any():
            mask = np.zeros(len(trials), bool); mask[0] = True
        mean_abs = float(np.abs(rpe[mask, stim]).mean())
        out.setdefault(mt, []).append((seed, ttr, mean_abs))
    return out


def fig_rpe_vs_recovery_time(data, out, stim_label="0%→100%", window_trials=500,
                             fname="rpe_vs_recovery_time"):
    """Scatter: mean early-reversal |RPE| (y) vs. trials-to-recovery (x), one point per
    RECOVERED seed, coloured by model class. Per-class Pearson r (p) annotated directly
    in the legend; pooled Pearson + Spearman r (p) in a text box on the axes. Full
    per-class Spearman also printed to console (both r and p already on-figure for
    Pearson, the primary statistic; Spearman is the robustness check)."""
    from scipy.stats import pearsonr, spearmanr
    types = [m for m in F.MODELS if m in data and data[m]]
    if not types:
        print(f"  (skip {fname}: no data)"); return

    # -- compute all stats first so they can be annotated on the figure, not just printed
    stats = {}
    for mt in types:
        rows = data[mt]
        ttrs = np.array([r[1] for r in rows]); rpes = np.array([r[2] for r in rows])
        if len(rows) >= 3:
            r, p = pearsonr(ttrs, rpes); rs, ps = spearmanr(ttrs, rpes)
        else:
            r = p = rs = ps = float("nan")
        stats[mt] = dict(ttrs=ttrs, rpes=rpes, n=len(rows), r=r, p=p, rs=rs, ps=ps)
    all_ttr = np.concatenate([stats[m]["ttrs"] for m in types])
    all_rpe = np.concatenate([stats[m]["rpes"] for m in types])
    if len(all_ttr) >= 3:
        pooled_r, pooled_p = pearsonr(all_ttr, all_rpe)
        pooled_rs, pooled_ps = spearmanr(all_ttr, all_rpe)
    else:
        pooled_r = pooled_p = pooled_rs = pooled_ps = float("nan")

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    for mt in types:
        s = stats[mt]
        r_txt = f"r={s['r']:+.2f}, p={s['p']:.3f}" if np.isfinite(s["r"]) else "n too small"
        ax.scatter(s["ttrs"], s["rpes"], color=F.MODELS[mt]["color"], s=46, edgecolor="white",
                  linewidth=0.6, alpha=0.85,
                  label=f"{F.MODELS[mt]['label']} (n={s['n']}): {r_txt}", zorder=3)
    ax.set_xlabel("trials to recovery (reward-based)", fontsize=13)
    ax.set_ylabel(f"mean |RPE|, stim {stim_label}\n(first {window_trials} reversal trials)",
                  fontsize=13)
    ax.tick_params(labelsize=11)
    ax.set_title("Early reward-prediction error vs. speed of recovery\n"
                 "(recovered seeds only -- TTR undefined otherwise)", fontsize=12)
    ax.legend(fontsize=8.5, loc="upper right")
    if np.isfinite(pooled_r):
        ax.text(0.02, 0.03,
                f"all classes pooled (n={len(all_ttr)}):\n"
                f"pearson r={pooled_r:+.2f} (p={pooled_p:.3f})\n"
                f"spearman r={pooled_rs:+.2f} (p={pooled_ps:.3f})",
                transform=ax.transAxes, fontsize=9, va="bottom", ha="left",
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7", alpha=0.9))
    fig.tight_layout(); F.save_fig(fig, out, fname)

    print("\n  Early |RPE| vs. trials-to-recovery correlation (recovered seeds only):")
    for mt in types:
        s = stats[mt]
        if s["n"] < 3:
            print(f"    {F.MODELS[mt]['label']:32s} n={s['n']} (too few for correlation)")
            continue
        print(f"    {F.MODELS[mt]['label']:32s} n={s['n']:2d}  "
              f"pearson r={s['r']:+.2f} (p={s['p']:.3f})  "
              f"spearman r={s['rs']:+.2f} (p={s['ps']:.3f})")
    if np.isfinite(pooled_r):
        print(f"    {'ALL CLASSES POOLED':32s} n={len(all_ttr):2d}  "
              f"pearson r={pooled_r:+.2f} (p={pooled_p:.3f})  "
              f"spearman r={pooled_rs:+.2f} (p={pooled_ps:.3f})")


def rpe_vs_recovery_time_per_stim(pre_runs, post_runs, stim, thr=0.2, smooth_window=3,
                                  window_trials=500):
    """Like rpe_vs_recovery_time, but BOTH sides of the pairing are specific to a single
    stimulus: TTR comes from time_to_recovery_vigour_target_per_stim (the reward-based
    TTR used by rpe_vs_recovery_time can't be split by stimulus -- see that function's
    docstring), and the early-RPE window is that same stimulus's own probe_rpe trace.
    {model_type: [(seed, ttr, mean_abs_rpe_early)]}, one row per seed that reached
    vigour target-match recovery FOR THIS STIMULUS specifically (a seed can recover on
    stim 0 but not stim 2, or vice versa -- that asymmetry is exactly the point)."""
    ttr_table = time_to_recovery_vigour_target_per_stim(pre_runs, post_runs, thr=thr,
                                                        smooth_window=smooth_window,
                                                        stim=stim)
    out = {}
    for f in glob.glob(str(Path(post_runs) / "*" / "seed*" / "history.json")):
        p = Path(f); mt = p.parent.parent.name; seed = int(p.parent.name[4:])
        if mt not in ttr_table or seed not in ttr_table[mt]:
            continue
        ttr = ttr_table[mt][seed]
        if not np.isfinite(ttr):
            continue
        h = json.loads(p.read_text())
        pu, rpe = h.get("probe_update"), h.get("probe_rpe")
        if not pu or not rpe:
            continue
        tot = max(h.get("total_updates", len(pu)), 1); ntr = h.get("n_trials", 2500)
        trials = np.asarray(pu, float) * ntr / tot
        rpe = np.asarray(rpe, float)                    # (n_probes, 3)
        mask = trials <= window_trials
        if not mask.any():
            mask = np.zeros(len(trials), bool); mask[0] = True
        mean_abs = float(np.abs(rpe[mask, stim]).mean())
        out.setdefault(mt, []).append((seed, ttr, mean_abs))
    return out


def fig_rpe_vs_recovery_time_per_stim(data0, data2, out, window_trials=500,
                                      fname="rpe_vs_recovery_time_per_stim"):
    """Two-panel version of fig_rpe_vs_recovery_time: LEFT = stim 0 (0%->100%, newly
    VALUABLE), RIGHT = stim 2 (100%->0%, newly WORTHLESS), each using that stimulus's
    OWN vigour target-match recovery time (not the whole-seed reward-based TTR, which
    isn't stimulus-separable -- see rpe_vs_recovery_time_per_stim's docstring), so the
    two directions of the reversal can be compared directly: is relearning the newly
    valuable stimulus faster or slower than relearning the newly worthless one? Per-
    class + pooled Pearson/Spearman annotated on each panel, same style as the combined
    figure. Also prints each class's median TTR for both directions side by side, which
    is the most direct answer to "is 100->0 faster than 0->100" (the correlation stats
    answer a different question -- whether early RPE size predicts speed WITHIN a
    direction -- not which direction is faster overall)."""
    from scipy.stats import pearsonr, spearmanr
    panels = [("0% → 100%  (newly valuable)", data0), ("100% → 0%  (newly worthless)", data2)]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2), sharey=True)
    for ax, (title, data) in zip(axes, panels):
        types = [m for m in F.MODELS if m in data and data[m]]
        if not types:
            ax.set_title(f"{title}\n(no data)", fontsize=12); ax.axis("off"); continue
        all_ttr, all_rpe = [], []
        for mt in types:
            rows = data[mt]
            ttrs = np.array([r[1] for r in rows]); rpes = np.array([r[2] for r in rows])
            all_ttr.append(ttrs); all_rpe.append(rpes)
            if len(rows) >= 3:
                r, p = pearsonr(ttrs, rpes)
                r_txt = f"r={r:+.2f}, p={p:.3f}"
            else:
                r_txt = "n too small"
            ax.scatter(ttrs, rpes, color=F.MODELS[mt]["color"], s=46, edgecolor="white",
                      linewidth=0.6, alpha=0.85,
                      label=f"{F.MODELS[mt]['label']} (n={len(rows)}): {r_txt}", zorder=3)
        all_ttr = np.concatenate(all_ttr); all_rpe = np.concatenate(all_rpe)
        if len(all_ttr) >= 3:
            pr, pp = pearsonr(all_ttr, all_rpe); prs, pps = spearmanr(all_ttr, all_rpe)
            ax.text(0.02, 0.03, f"pooled (n={len(all_ttr)}):\npearson r={pr:+.2f} "
                    f"(p={pp:.3f})\nspearman r={prs:+.2f} (p={pps:.3f})",
                    transform=ax.transAxes, fontsize=8.5, va="bottom", ha="left",
                    bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7",
                             alpha=0.9))
        ax.set_xlabel("trials to recovery\n(vigour target-match, this stimulus)", fontsize=12)
        ax.tick_params(labelsize=10)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=7.5, loc="upper right")
    axes[0].set_ylabel(f"mean |RPE|, this stimulus\n(first {window_trials} reversal trials)",
                       fontsize=13)
    fig.suptitle("Early RPE vs. speed of recovery, split by direction of value change\n"
                "(recovered seeds only per panel -- TTR undefined otherwise)", fontsize=12)
    fig.tight_layout(); F.save_fig(fig, out, fname)

    print("\n  Median trials-to-recovery by direction (vigour target-match, per stimulus):")
    types = sorted(set(list(data0.keys()) + list(data2.keys())),
                   key=lambda m: list(F.MODELS).index(m) if m in F.MODELS else 99)
    for mt in types:
        label = F.MODELS[mt]["label"] if mt in F.MODELS else mt
        m0 = np.median([r[1] for r in data0.get(mt, [])]) if data0.get(mt) else float("nan")
        m2 = np.median([r[1] for r in data2.get(mt, [])]) if data2.get(mt) else float("nan")
        n0, n2 = len(data0.get(mt, [])), len(data2.get(mt, []))
        diff = f"  ({'0->100 faster' if m0 < m2 else '100->0 faster'} by {abs(m0 - m2):.0f})" \
               if np.isfinite(m0) and np.isfinite(m2) else ""
        print(f"    {label:32s} 0%→100%: {m0:6.0f} (n={n0:2d})   "
              f"100%→0%: {m2:6.0f} (n={n2:2d}){diff}")


def fig_reward_scale_paired(pre_runs, panels, out, thr=0.2, smooth_window=3,
                            fname="reward_scale_intervention_paired"):
    """Paired per-seed slope plot for the reward_scale_by_stim causal intervention (see
    run_reward_scale_intervention.sh): one line per seed, per model class, connecting
    that seed's control-condition TTR to its treatment-condition TTR (vigour target-
    match, stimulus-specific). Because control/treatment runs share seeds AND matched
    RNG draws (same base_seed/model_seed -> same env trial sequence and same sampled
    policy noise -- only the reward_scale differs), a line's slope isolates the
    intervention's effect from seed-to-seed noise far better than comparing unpaired
    medians would. Thick marker-line = across-seed median.

    panels: list of (title, control_runs, treat_runs, stim) tuples -- typically the
    TARGETED stimulus (effect predicted) and the control-check stimulus (untouched,
    should show ~no consistent effect) side by side, for one or more conditions."""
    types = list(F.MODELS)
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n + 1, 5.6), squeeze=False)
    axes = axes[0]
    for ax, (title, control_runs, treat_runs, stim) in zip(axes, panels):
        ctl = time_to_recovery_vigour_target_per_stim(pre_runs, control_runs, thr=thr,
                                                       smooth_window=smooth_window, stim=stim)
        trt = time_to_recovery_vigour_target_per_stim(pre_runs, treat_runs, thr=thr,
                                                       smooth_window=smooth_window, stim=stim)
        for mt in types:
            if mt not in ctl or mt not in trt:
                continue
            common = sorted(set(ctl[mt]) & set(trt[mt]))
            pairs = [(ctl[mt][s], trt[mt][s]) for s in common
                    if np.isfinite(ctl[mt][s]) and np.isfinite(trt[mt][s])]
            if not pairs:
                continue
            col = F.MODELS[mt]["color"]
            for c, t in pairs:
                ax.plot([0, 1], [c, t], color=col, alpha=0.35, lw=1.1, zorder=2)
            cs, ts = zip(*pairs)
            ax.plot([0, 1], [np.median(cs), np.median(ts)], color=col, lw=3.2, marker="o",
                    markersize=7, zorder=4, label=f"{F.MODELS[mt]['label']} (n={len(pairs)})")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["control", "treatment"], fontsize=10)
        ax.set_xlim(-0.35, 1.35)
        ax.set_title(title, fontsize=11)
        ax.tick_params(axis="y", labelsize=10)
        if ax is axes[0]:
            ax.set_ylabel("trials to recovery\n(vigour target-match, this stimulus)", fontsize=12)
        ax.legend(fontsize=7.5, loc="best")
    fig.suptitle("Reward-scale causal intervention: paired per-seed effect\n"
                "(same seed = identical RNG draws + env in both conditions)", fontsize=12)
    fig.tight_layout(); F.save_fig(fig, out, fname)


def fig_reward_scale_dose_response(pre_runs, scale_runs, stim, out,
                                   thr=0.2, smooth_window=3, xlabel="reward scale",
                                   fname="reward_scale_dose_response"):
    """Dose-response curve: x = reward_scale_by_stim's multiplier for `stim` (e.g. 0.1,
    0.3, 0.5, 0.7, 1.0=control), y = per-seed vigour target-match TTR for that SAME
    stimulus, one line per model class connecting the across-seed median at each dose
    (thin dots = individual seeds, jittered in x). Requires every dose's run dir to
    share seeds/RNG with the others (see run_reward_scale_dose_response.sh) for the
    same paired-noise-reduction reason as fig_reward_scale_paired.

    scale_runs: list of (scale_value, runs_dir) tuples, sorted ascending by scale_value
    by this function (caller doesn't need to pre-sort)."""
    scale_runs = sorted(scale_runs, key=lambda x: x[0])
    types = list(F.MODELS)
    tables = [(sv, time_to_recovery_vigour_target_per_stim(pre_runs, rd, thr=thr,
                                                           smooth_window=smooth_window, stim=stim))
             for sv, rd in scale_runs]
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    rng = np.random.default_rng(0)
    for mt in types:
        if not any(mt in t for _, t in tables):
            continue
        xs_med, ys_med = [], []
        for sv, t in tables:
            d = t.get(mt, {})
            vals = np.array([v for v in d.values() if np.isfinite(v)])
            if len(vals) == 0:
                continue
            jitter = rng.uniform(-0.02, 0.02, size=len(vals)) * max(sv, 0.1)
            ax.scatter(np.full(len(vals), sv) + jitter, vals, color=F.MODELS[mt]["color"],
                      s=22, alpha=0.4, zorder=2)
            xs_med.append(sv); ys_med.append(np.median(vals))
        if xs_med:
            ax.plot(xs_med, ys_med, color=F.MODELS[mt]["color"], lw=2.6, marker="o",
                    markersize=8, zorder=4, label=F.MODELS[mt]["label"])
    ax.axvline(1.0, color="0.5", lw=1.2, ls="--", zorder=1)
    ax.text(1.0, ax.get_ylim()[1], " unscaled\n control", fontsize=8, color="0.4",
           va="top", ha="left")
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel("trials to recovery\n(vigour target-match, this stimulus)", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.set_title("Dose-response: reward-scale magnitude vs. relearning speed", fontsize=12)
    ax.legend(fontsize=9.5, loc="best")
    fig.tight_layout(); F.save_fig(fig, out, fname)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre-runs", default="results/model_runs")
    ap.add_argument("--post-runs", default="results/reversal_5000/model_runs_reversal")
    ap.add_argument("--out", default="results/reversal_5000/figures_seed_groups")
    ap.add_argument("--thr", type=float, default=0.8)
    ap.add_argument("--reward-smooth", type=int, default=51,
                    help="rolling-average window in UPDATES (mean_reward is logged "
                         "every update) for the reward-based crossing")
    ap.add_argument("--vigour-smooth", type=int, default=3,
                    help="rolling-average window in PROBES (probe_vigour is logged "
                         "every --probe-every updates) for the vigour-based crossing")
    ap.add_argument("--target-thr", type=float, default=0.2,
                    help="for the vigour target-matching definition: recovered once mean "
                         "|post-vigour - mirror-swapped pre-reversal target| across the 3 "
                         "stimuli drops to <= this fraction of the seed's own pre-reversal "
                         "reversed-value span")
    ap.add_argument("--style", default=str(F.DEFAULT_STYLE))
    args = ap.parse_args()
    matplotlib.use("Agg")
    if Path(args.style).exists():
        plt.style.use(args.style); print(f"style: {args.style}")
    else:
        print(f"** WARNING: style file not found at {args.style} — figures will use "
              f"default matplotlib styling, not the house style **")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    reward_ttr = time_to_recovery_reward(args.post_runs, thr=args.thr,
                                         smooth_window=args.reward_smooth)
    vigour_ttr = time_to_recovery_vigour(args.pre_runs, args.post_runs, thr=args.thr,
                                         smooth_window=args.vigour_smooth)
    target_ttr = time_to_recovery_vigour_target(args.pre_runs, args.post_runs,
                                                thr=args.target_thr,
                                                smooth_window=args.vigour_smooth)
    for mt in F.MODELS:
        for label, ttr in (("reward", reward_ttr), ("vigour", vigour_ttr),
                          ("target", target_ttr)):
            if mt in ttr:
                vals = [v for v in ttr[mt].values() if np.isfinite(v)]
                med = f"median={np.median(vals):.0f} trials" if vals else "no crossings"
                print(f"  {F.MODELS[mt]['label']:32s} {label:7s} TTR: "
                      f"n={len(vals)}/{len(ttr[mt])}  {med}")
    fig_recovery_time([
        ("reward-based", reward_ttr, f"{int(args.thr*100)}%"),
        ("vigour-based\n(reversed-span)", vigour_ttr, f"{int(args.thr*100)}%"),
        ("vigour-based\n(target-match)", target_ttr, f"within {int(args.target_thr*100)}%"),
    ], out)

    reward_table = SG.load_recovery_table(args.post_runs)
    fig_recovery_definition_comparison(reward_table, args.thr, vigour_ttr, target_ttr,
                                       out, args.target_thr)

    rpe_data = rpe_vs_recovery_time(args.post_runs, thr=args.thr,
                                    reward_smooth=args.reward_smooth)
    fig_rpe_vs_recovery_time(rpe_data, out)

    rpe_data_0 = rpe_vs_recovery_time_per_stim(args.pre_runs, args.post_runs, stim=0,
                                               thr=args.target_thr,
                                               smooth_window=args.vigour_smooth)
    rpe_data_2 = rpe_vs_recovery_time_per_stim(args.pre_runs, args.post_runs, stim=2,
                                               thr=args.target_thr,
                                               smooth_window=args.vigour_smooth)
    fig_rpe_vs_recovery_time_per_stim(rpe_data_0, rpe_data_2, out)


if __name__ == "__main__":
    main()
