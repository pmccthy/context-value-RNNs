"""coding_angle_comparison.py

Second-order similarity: how well does each RNN model type's combined
pre+post coding-angle (RSA) matrix -- see analysis/coding_angle.py's
combined_coding_angle_summary() -- match the real experimental one (see
~/Documents/neuronal-representations/results/transfer/figures/panels/
coding_angle_experimental.py's coding_angle_summary_experimental())?

This is a comparison of two 6x6 *similarity* matrices (RSA-of-an-RSA /
"second-order isomorphism"), not of the underlying coding vectors
themselves -- appropriate here because, exactly as documented in
coding_angle_experimental.py's docstring, neuron identity does not
correspond across RNN seeds or across experimental sessions, so only the
scalar cosine-similarity VALUES (not the raw vectors) are ever comparable
across those units. The 15 off-diagonal upper-triangle entries of each
6x6 matrix (diagonal is trivially 1.0 by construction on both sides) are
extracted, the two label orderings are aligned by CONDITION IDENTITY (not
position -- the RNN and experimental sides list the 0%->100% / 100%->0%
stem-pair transitions in opposite order, and use different arrow glyphs,
see _canon() below), and compared with Pearson (and Spearman, rank-based)
correlation.

Per the cross-repo "no cross-import" convention used throughout this
directory (see README.md and e.g. ../panels/reversal_gradient_bars.py's
vendored _exp_reversal_points()), the experimental coding-angle summary is
NOT imported from the neuronal-representations repo's panels module --
it is recomputed here directly from the same raw *_tdr_v2.pkl files
(EXPERIMENTAL_RESULTS_DIR), duplicating coding_angle_experimental.py's
_all_session_matrices()/coding_angle_summary_experimental() logic. Keep
the two in sync by hand if the experimental-side computation ever changes.
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "panels"))
sys.path.insert(0, str(_HERE.parent / "style"))
sys.path.insert(0, str(_HERE.parent / "transfer" / "code"))
sys.path.insert(0, str(_HERE.parent / "analysis"))

from _tags import new_panel, save_panel  # noqa: E402
import coding_angle as CANG  # noqa: E402
import figures as F  # noqa: E402

MODEL_TYPES = ["rl_only", "classif_rl", "classif_rl_readout_only"]

# ---------------------------------------------------------------------------
# Experimental side -- vendored from coding_angle_experimental.py (see
# module docstring for why this is duplicated rather than imported).
# ---------------------------------------------------------------------------
DEFAULT_RESULTS_DIR = "~/Documents/experimental_results/08_04_26_dff_from_flu_processed_zscore_snr_tdr_v2"
EXPERIMENTAL_RESULTS_DIR = Path(
    os.environ.get("EXPERIMENTAL_RESULTS_DIR", DEFAULT_RESULTS_DIR)
).expanduser()

_EXP_LABELS = ["100->0% (pre)", "50% (pre)", "0->100% (pre)",
               "100->0% (post)", "50% (post)", "0->100% (post)"]

_EXP_CACHE: dict = {}


def _exp_session_files():
    return sorted(EXPERIMENTAL_RESULTS_DIR.glob("*_tdr_v2.pkl"))


def _exp_condition_coding_vectors(d: dict) -> np.ndarray:
    fs = d["fs_image"]
    onset = d["stim_onset_abs"]
    win = slice(onset, onset + int(round(d["stim_duration_s"] * fs)))
    vecs = d["cond_avg"][:, :, win].mean(axis=2)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return vecs / norms


def _exp_all_session_matrices():
    if "mats" not in _EXP_CACHE:
        mats = []
        for f in _exp_session_files():
            with open(f, "rb") as fh:
                d = pickle.load(fh)
            vecs = _exp_condition_coding_vectors(d)
            mats.append(vecs @ vecs.T)
        if not mats:
            raise FileNotFoundError(f"no *_tdr_v2.pkl found under {EXPERIMENTAL_RESULTS_DIR}")
        _EXP_CACHE["mats"] = np.asarray(mats)
    return _EXP_CACHE["mats"]


def experimental_coding_angle_summary():
    """(mean (6,6), sem (6,6), n_sessions, labels) -- mirrors
    coding_angle_experimental.coding_angle_summary_experimental() plus the
    fixed label list, so callers here don't need the labels hardcoded twice."""
    mats = _exp_all_session_matrices()
    mean = mats.mean(axis=0)
    sem = mats.std(axis=0, ddof=1) / np.sqrt(mats.shape[0])
    return mean, sem, mats.shape[0], list(_EXP_LABELS)


# ---------------------------------------------------------------------------
# Label alignment: RNN and experimental sides both list [0%->100%, 50%,
# 100%->0%] x [pre, post] but in a DIFFERENT stem-pair order (and using
# different arrow glyphs -- unicode "->" vs ascii "->"), so any comparison
# must align by condition identity, not position.
# ---------------------------------------------------------------------------

def _canon(label: str) -> str:
    """Map a coding-angle condition label (either side's format) to a
    canonical "{transition}_{phase}" key, e.g. "0to100_pre". Identifies the
    transition category purely from the FIRST digit token (stripped of any
    '%'), since the two sides differ both in stem-pair order and arrow glyph
    but always list the FROM-value first: "0%\u2192100%..." / "0->100%..."
    both start with "0"; "100%\u21920%..." / "100->0%..." both start with
    "100"; "50%..." starts with "50"."""
    phase = "post" if "post" in label else "pre"
    stripped = label.replace("%", "").strip()
    if stripped.startswith("50"):
        trans = "50"
    elif stripped.startswith("100"):
        trans = "100to0"
    elif stripped.startswith("0"):
        trans = "0to100"
    else:
        raise ValueError(f"cannot canonicalise coding-angle label {label!r}")
    return f"{trans}_{phase}"
    return f"{trans}_{phase}"


def _align_to(labels_target, labels_source):
    """Index array `perm` such that reindexing a `labels_source`-ordered
    matrix with np.ix_(perm, perm) yields it in `labels_target`'s order."""
    canon_target = [_canon(l) for l in labels_target]
    canon_source = [_canon(l) for l in labels_source]
    if sorted(canon_target) != sorted(canon_source):
        raise ValueError(f"label sets don't match: {canon_target} vs {canon_source}")
    return [canon_source.index(c) for c in canon_target]


def _upper_tri_values(mat: np.ndarray) -> np.ndarray:
    """The 15 off-diagonal upper-triangle entries of a 6x6 matrix (the
    diagonal is trivially 1.0 by construction on both sides, so it carries
    no information and would only inflate the correlation)."""
    n = mat.shape[0]
    iu = np.triu_indices(n, k=1)
    return mat[iu]


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def second_order_similarity(model_type: str, seeds=None):
    """Pearson + Spearman correlation between `model_type`'s combined
    pre+post coding-angle matrix and the experimental one, over the 15
    off-diagonal upper-triangle entries (aligned by condition identity).
    Returns a dict with the two correlations, their sample sizes, and the
    two aligned value vectors (for a scatter plot if wanted)."""
    rnn_mean, _, n_seeds, rnn_labels = CANG.combined_coding_angle_summary(model_type, seeds=seeds)
    exp_mean, _, n_sessions, exp_labels = experimental_coding_angle_summary()

    perm = _align_to(rnn_labels, exp_labels)
    exp_aligned = exp_mean[np.ix_(perm, perm)]

    rnn_vals = _upper_tri_values(rnn_mean)
    exp_vals = _upper_tri_values(exp_aligned)

    try:
        from scipy.stats import pearsonr, spearmanr
        pear_r, pear_p = pearsonr(rnn_vals, exp_vals)
        spear_r, spear_p = spearmanr(rnn_vals, exp_vals)
    except ImportError:
        pear_r = float(np.corrcoef(rnn_vals, exp_vals)[0, 1])
        pear_p = spear_r = spear_p = float("nan")

    return {
        "model_type": model_type,
        "pearson_r": float(pear_r), "pearson_p": float(pear_p),
        "spearman_r": float(spear_r), "spearman_p": float(spear_p),
        "n_seeds": n_seeds, "n_sessions": n_sessions,
        "rnn_vals": rnn_vals, "exp_vals": exp_vals,
    }


def all_second_order_similarities(seeds=None):
    return [second_order_similarity(mt, seeds=seeds) for mt in MODEL_TYPES]


def second_order_similarity_per_session(model_type: str, seeds=None):
    """Per-EXPERIMENTAL-SESSION analogue of second_order_similarity: instead
    of correlating the model's RSA matrix against the session-AVERAGED
    experimental matrix (as draw_comparison_bar / second_order_similarity
    do), correlates it against each individual session's own (single-
    session, noisier) matrix -- one r per session. This is what
    session-to-session variability in the model-vs-experiment comparison
    actually looks like; it is NOT what draw_comparison_bar's error bars
    show (those are an analytic CI on the session-averaged comparison, see
    that function's docstring). Uses the same model-mean RSA matrix
    throughout -- the model side has no natural "single session" of its
    own (it has seeds, which are a different kind of unit, aggregated
    separately elsewhere) -- only the experimental side is disaggregated
    here."""
    rnn_mean, _, n_seeds, rnn_labels = CANG.combined_coding_angle_summary(model_type, seeds=seeds)
    exp_mats = _exp_all_session_matrices()
    _, _, _, exp_labels = experimental_coding_angle_summary()

    perm = _align_to(rnn_labels, exp_labels)
    rnn_vals = _upper_tri_values(rnn_mean)

    rs = []
    for mat in exp_mats:
        aligned = mat[np.ix_(perm, perm)]
        exp_vals = _upper_tri_values(aligned)
        rs.append(float(np.corrcoef(rnn_vals, exp_vals)[0, 1]))
    return {"model_type": model_type, "n_seeds": n_seeds,
            "n_sessions": len(rs), "session_r": rs}


def draw_comparison_per_session(ax=None):
    """Standalone panel (NOT wired into the MODELFIG2 composite, on
    request -- see draw_comparison_bar for the composite-embedded summary
    version) showing session-to-session variability in the model-vs-
    experiment RSA-matrix correlation: for each model type, one Pearson r
    per experimental session (second_order_similarity_per_session), shown
    as jittered individual points, with the across-session mean +/- SEM
    overlaid as a marker+errorbar."""
    fig, ax, _ = new_panel(ax, figsize=(5.0, 4.5))
    rng = np.random.default_rng(0)
    n_sessions = 0
    for i, mt in enumerate(MODEL_TYPES):
        res = second_order_similarity_per_session(mt)
        rs = np.asarray(res["session_r"])
        n_sessions = res["n_sessions"]
        colour = F.MODELS[mt]["color"]
        jitter = rng.uniform(-0.12, 0.12, size=len(rs))
        ax.scatter(np.full(len(rs), i) + jitter, rs, color=colour, s=18, alpha=0.6, zorder=2)
        m, sem = rs.mean(), rs.std(ddof=1) / np.sqrt(len(rs))
        ax.errorbar([i], [m], yerr=[sem], color=colour, marker="_", ms=22, mew=2.5,
                    lw=0, elinewidth=2.0, capsize=5, zorder=3)
    ax.axhline(0, color="0.4", lw=0.8, zorder=1)
    ax.set_xticks(range(len(MODEL_TYPES)))
    ax.set_xticklabels([F.MODELS[m]["label"] for m in MODEL_TYPES], fontsize=10)
    ax.set_ylabel("Pearson r, model vs.\none experimental session's RSM")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Session-to-session variability,\nmodel-vs-experiment RSA correlation\n"
                 f"[points = individual sessions, n={n_sessions}; bar = mean +/- SEM]", fontsize=10)
    return fig


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

def draw_comparison_bar(ax=None, show_errorbars=True, ylim=(-1.0, 1.0)):
    """One bar per model type: Pearson r between that model's combined
    coding-angle matrix and the experimental one (over the 15 off-diagonal
    upper-triangle entries, aligned by condition identity). Error shown
    (when show_errorbars=True, the default) is the 95% CI of r from
    Fisher's z-transform (n = 15 matrix entries, not seeds/sessions -- this
    is a correlation over CONDITION PAIRS, so that's the correct n for the
    CI, not the seed/session counts also reported). show_errorbars=False
    plots the same bars with no yerr, e.g. for a cleaner summary view
    where the CI isn't the point being made. ylim defaults to the full
    [-1, 1] correlation range (the axhline(0) reference line only makes
    sense with negative values in view); pass (0.0, 1.0) for a tighter
    view when every bar is known to be positive, as here."""
    fig, ax, _ = new_panel(ax, figsize=(5.0, 4.5))
    results = all_second_order_similarities()

    xs = np.arange(len(results))
    rs = [r["pearson_r"] for r in results]
    ns = [len(r["rnn_vals"]) for r in results]
    yerr = None
    if show_errorbars:
        los, his = [], []
        for r, n in zip(rs, ns):
            r_c = float(np.clip(r, -0.9999, 0.9999))
            z = np.arctanh(r_c)
            se = 1.0 / np.sqrt(max(n - 3, 1))
            lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
            los.append(r - lo); his.append(hi - r)
        yerr = [los, his]

    colours = [F.MODELS[m]["color"] for m in MODEL_TYPES]
    labels = [F.MODELS[m]["label"] for m in MODEL_TYPES]
    ax.bar(xs, rs, width=0.6, color=colours, yerr=yerr, capsize=4, zorder=2)
    ax.axhline(0, color="0.4", lw=0.8, zorder=1)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Pearson r (model vs. exp.)")
    ax.set_ylim(*ylim)
    ax.spines[["top", "right"]].set_visible(False)
    n_sessions = results[0]["n_sessions"] if results else 0
    ax.set_title("Coding-angle matrix similarity to experimental data\n"
                 f"[15 condition pairs; n={n_sessions} sessions]", fontsize=10)
    return fig


def draw_experimental_heatmap_triangular(ax=None, colorbar=None):
    """Experimental-side combined 6-condition (3 stem-pairs x pre/post)
    coding-angle heatmap, triangular (redundant upper half masked out) --
    vendored from neuronal-representations' coding_angle_experimental.
    draw_coding_angle_heatmap_triangular() per this directory's own
    "no cross-import" convention (see module docstring): recomputed here
    from experimental_coding_angle_summary() above rather than imported, so
    keep the two in sync by hand if the experimental-side computation ever
    changes. Styling matches coding_angle_panel.draw_coding_angle_heatmap_
    combined (sequential white->red [0,1] colormap) so the two are visually
    comparable apples-to-apples when placed side by side in a composite."""
    fig, ax, owns = new_panel(ax, figsize=(5.2, 5.2))
    if colorbar is None:
        colorbar = owns
    mean, sem, n_sessions, labels = experimental_coding_angle_summary()
    n = len(labels)
    hide = np.triu(np.ones((n, n), dtype=bool), k=1)
    cmap = plt.get_cmap("Reds").copy()
    cmap.set_bad(color="white")
    masked = np.ma.masked_where(hide, mean)
    im = ax.imshow(masked, vmin=0, vmax=1, cmap=cmap)
    for i in range(n):
        for j in range(n):
            if hide[i, j]:
                continue
            txt = f"{mean[i, j]:.2f}" if i != j else "1.00"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                     color="white" if abs(mean[i, j]) > 0.5 else "black")
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=7)
    ax.set_ylabel("cosine similarity")
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="cosine similarity")
    ax.set_title(f"Neural data\n[n={n_sessions} sessions]", fontsize=9)
    return fig


def build_all(show_tag=None):
    try:
        fig = draw_comparison_bar()
    except FileNotFoundError as e:
        print(f"  (skip coding-angle model-vs-experiment comparison: {e})")
        return
    save_panel(fig, "Mechanistic/CodingAngle", "MECH.ANGLE.vsexp",
               "coding_angle_vs_experimental", show_tag)


if __name__ == "__main__":
    for r in all_second_order_similarities():
        print(f"{r['model_type']:28s}  pearson r={r['pearson_r']:+.3f} (p={r['pearson_p']:.4g})"
              f"   spearman r={r['spearman_r']:+.3f} (p={r['spearman_p']:.4g})"
              f"   n_seeds={r['n_seeds']}  n_sessions={r['n_sessions']}")
    build_all()
