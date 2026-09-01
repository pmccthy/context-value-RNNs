"""Alpha-sweep diagnostic: how do responder-significance rates change with the
p-value threshold, for Method A (paired window-mean test, our production
method) vs Method B (temporal per-timepoint test), with and without
across-unit FDR correction? One model type, all seeds pooled, pre-reversal.

Directly answers: (1) does Method A show the same large-N blowup as Method B
if you push alpha hard enough / leave it lenient? (2) does FDR correction fix
Method B's over-significance at the usual alpha=0.05?
"""
import glob, re, sys
from pathlib import Path
import numpy as np
from scipy.stats import ttest_1samp, ttest_ind
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_TYPE = "classif_rl"
TR_DIR = Path("transfer/figure_data/time_resolved")
ALPHAS = np.logspace(-10, np.log10(0.5), 14)

files = sorted(glob.glob(str(TR_DIR / f"{MODEL_TYPE}_seed*.npz")))
print(f"{len(files)} seeds for {MODEL_TYPE}")

# ---- collect raw p-values, ONCE, for both methods (no threshold applied yet) ----
pA = []   # list of (H,) one-sided p-vals, per seed per stim -> will stack to (n_seed*H, 3)... 
          # actually keep per-stim lists separately: pA[s] = list of (H,) arrays across seeds
pA_by_stim = {0: [], 1: [], 2: []}
pB_by_stim = {0: [], 1: [], 2: []}   # native-variant per-timepoint p-vals, (stim_ts, H) per seed

for f in files:
    z = np.load(f, allow_pickle=True)
    aligned = np.asarray(z["aligned"], dtype=np.float32)   # (trial, time, unit)
    stim = np.asarray(z["stimulus"]).astype(int)
    n_iti, stim_ts, rew_ts = (int(b) for b in z["bounds"])
    R = aligned[:, n_iti:n_iti + stim_ts, :].mean(1)        # (trial, unit)
    Bl = aligned[:, 0:n_iti, :].mean(1)
    delta = R - Bl
    for s in range(3):
        idx = stim == s
        # Method A: single paired t-test, window means
        t, p = ttest_1samp(delta[idx], 0.0, axis=0)
        p_one = np.where(t > 0, p / 2, 1.0)
        p_one = np.nan_to_num(p_one, nan=1.0)
        pA_by_stim[s].append(p_one)
        # Method B (native): per-timepoint independent t-test vs pooled baseline
        baseline = aligned[idx, 0:n_iti, :]                        # (n_trial_s, n_iti, H)
        baseline_pooled = baseline.reshape(-1, baseline.shape[-1])  # (n_trial_s*n_iti, H)
        stim_win = aligned[idx, n_iti:n_iti + stim_ts, :]           # (n_trial_s, stim_ts, H)
        p_vals = np.zeros((stim_ts, baseline.shape[-1]))
        for t_ in range(stim_ts):
            _, p_t = ttest_ind(stim_win[:, t_, :], baseline_pooled, axis=0)
            p_vals[t_] = np.where(np.isnan(p_t), 1.0, p_t)
        pB_by_stim[s].append(p_vals)

for s in range(3):
    pA_by_stim[s] = np.concatenate(pA_by_stim[s])                    # (n_seed*H,)
    pB_by_stim[s] = np.stack(pB_by_stim[s], axis=-1)                 # (stim_ts, H, n_seed)

def length_of_continuous_subarray(arr, max_int=3):
    def longest_false_streak(a):
        max_len = current = 0
        for val in a:
            if not val:
                current += 1; max_len = max(max_len, current)
            else:
                current = 0
        return max_len
    arr = np.asarray(arr)
    if arr.size < 2:
        return 0
    iv = np.diff(arr)
    return longest_false_streak(np.diff(iv <= max_int))

def sigB_from_pvals(p_vals_hxT, alpha, correct, max_int=3, frac=1/3):
    # p_vals_hxT: (stim_ts, H) for ONE seed
    stim_ts, H = p_vals_hxT.shape
    pv = p_vals_hxT.copy()
    if correct:
        for h in range(H):
            _, pv[:, h], _, _ = multipletests(pv[:, h], alpha=alpha, method="fdr_bh")
    thresh = frac * stim_ts
    sig = np.zeros(H, dtype=bool)
    for h in range(H):
        idx = np.where(pv[:, h] < alpha)[0]
        sig[h] = length_of_continuous_subarray(idx, max_int=max_int) > thresh
    return sig

results = {"alpha": ALPHAS.tolist()}
for corrA in (False, True):
    keyA = f"A_{'fdr' if corrA else 'raw'}"
    frac_sig, frac_all3 = [], []
    for a in ALPHAS:
        sig3 = []
        for s in range(3):
            p = pA_by_stim[s]
            if corrA:
                _, p, _, _ = multipletests(p, alpha=a, method="fdr_bh")
            sig3.append(p < a)
        sig3 = np.stack(sig3, axis=1)               # (n_seed*H, 3)
        frac_sig.append(float(sig3.mean()))
        frac_all3.append(float(sig3.all(axis=1).mean()))
    results[keyA] = {"frac_sig": frac_sig, "frac_all3": frac_all3}
    print(keyA, "done")

n_seeds = pB_by_stim[0].shape[-1]
for corrB in (False, True):
    keyB = f"B_{'fdr' if corrB else 'raw'}"
    frac_sig, frac_all3 = [], []
    for a in ALPHAS:
        sig3 = []
        for s in range(3):
            per_seed_sig = []
            for si in range(n_seeds):
                per_seed_sig.append(sigB_from_pvals(pB_by_stim[s][:, :, si], a, corrB))
            sig3.append(np.concatenate(per_seed_sig))   # (n_seed*H,)
        sig3 = np.stack(sig3, axis=1)
        frac_sig.append(float(sig3.mean()))
        frac_all3.append(float(sig3.all(axis=1).mean()))
    results[keyB] = {"frac_sig": frac_sig, "frac_all3": frac_all3}
    print(keyB, "done")

import json
Path("output/analysis").mkdir(parents=True, exist_ok=True)
json.dump(results, open("output/analysis/alpha_sweep.json", "w"), indent=2)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
styles = {"A_raw": dict(color="#1f77b4", ls="-", label="Method A (window-mean), uncorrected"),
          "A_fdr": dict(color="#1f77b4", ls="--", label="Method A (window-mean), FDR-BH"),
          "B_raw": dict(color="#d62728", ls="-", label="Method B (temporal), uncorrected"),
          "B_fdr": dict(color="#d62728", ls="--", label="Method B (temporal), FDR-BH")}
for k, st in styles.items():
    axes[0].plot(ALPHAS, results[k]["frac_sig"], marker="o", ms=3, **st)
    axes[1].plot(ALPHAS, results[k]["frac_all3"], marker="o", ms=3, **st)
for ax, title in zip(axes, ["fraction of (unit, stimulus) tests significant",
                            "fraction of units significant for ALL 3 stimuli"]):
    ax.set_xscale("log"); ax.set_xlabel(r"$\alpha$ threshold"); ax.set_ylabel("fraction")
    ax.set_title(title, fontsize=10)
    ax.axvline(0.05, color="0.6", ls=":", lw=1)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].legend(frameon=False, fontsize=7, loc="upper left")
fig.suptitle(f"Responder-significance rate vs. alpha ({MODEL_TYPE}, pre-reversal, all seeds pooled)",
             fontsize=11)
fig.tight_layout()
fig.savefig("output/analysis/alpha_sweep.png", dpi=150)
print("saved output/analysis/alpha_sweep.png")
