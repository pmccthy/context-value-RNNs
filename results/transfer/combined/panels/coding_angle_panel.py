"""coding_angle.py (panel)

Draws the per-model_type coding-angle (cosine-similarity) structure between
stimulus conditions' average normalised coding vectors, and a cross-model-type
comparison bar chart of the same condition-pair similarities.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "style"))
sys.path.insert(0, str(_HERE.parent / "transfer" / "code"))
sys.path.insert(0, str(_HERE.parent / "analysis"))

from _tags import new_panel, save_panel  # noqa: E402
import figure_config as FC  # noqa: E402
import coding_angle as CA  # noqa: E402

MODEL_TYPES = ["rl_only", "classif_rl", "classif_rl_readout_only"]


def draw_coding_angle_heatmap(model_type, phase="pre", ax=None, colorbar=None):
    fig, ax, owns = new_panel(ax, figsize=(4.6, 4.6))
    # colorbar defaults to "standalone only" (owns==True): plt.colorbar() adds a
    # NEW axes to the figure, which silently breaks compose.py's _column_headers()
    # -- it assumes fig.axes[0:n_cols] are exactly the grid's own column axes, and a
    # stray colorbar axes inserted between columns shifts that indexing and misaligns
    # every header. Composites pass colorbar=False explicitly (or rely on this default).
    if colorbar is None:
        colorbar = owns
    mean, sem, n_seeds, labels = CA.coding_angle_summary(model_type, phase=phase)
    im = ax.imshow(mean, vmin=-1, vmax=1, cmap="RdBu_r")
    for i in range(len(labels)):
        for j in range(len(labels)):
            txt = f"{mean[i, j]:.2f}" if i != j else "1.00"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                     color="white" if abs(mean[i, j]) > 0.5 else "black")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="cosine similarity")
def draw_coding_angle_heatmap_combined(model_type, seeds=None, ax=None, colorbar=None):
    """Single combined pre+post-reversal coding-angle heatmap (6 conditions:
    3 pre-reversal stimuli then 3 post-reversal), replacing the two separate
    pre/post heatmaps. The matrix is exactly symmetric with an exactly-1.0
    diagonal by construction (see coding_angle.combined_coding_angle_summary),
    so only the lower triangle (incl. diagonal) is drawn -- the upper
    triangle is masked out (left white) since it's redundant."""
    fig, ax, owns = new_panel(ax, figsize=(6.2, 6.2))
    if colorbar is None:
        colorbar = owns
    mean, sem, n_seeds, labels = CA.combined_coding_angle_summary(model_type, seeds=seeds)
    n = len(labels)
    hide = np.triu(np.ones((n, n), dtype=bool), k=1)  # strictly-upper triangle only (keep diagonal + lower)
    cmap = plt.get_cmap("Reds").copy()  # sequential white->red -- values are all
    # positive in practice, so no need for a diverging colormap's blue half
    cmap.set_bad(color="white")
    masked = np.ma.masked_where(hide, mean)
    im = ax.imshow(masked, vmin=0, vmax=1, cmap=cmap)  # all-positive similarities in
    # practice -- crop to [0, 1] instead of [-1, 1] so the full colormap range is used
    for i in range(n):
        for j in range(n):
            if hide[i, j]:
                continue
            txt = f"{mean[i, j]:.2f}" if i != j else "1.00"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                     color="white" if abs(mean[i, j]) > 0.5 else "black")
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_ylabel("cosine similarity")
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="cosine similarity")
    mdl_label = FC.MODELS.get(model_type, {}).get("label", model_type)
    ax.set_title(f"{mdl_label} coding-vector angle, pre+post reversal combined\n"
                 f"(triangular -- symmetric by construction)  [n={n_seeds} seeds]")
    return fig


def draw_coding_angle_by_model(phase="pre", ax=None):
    """Bar chart: for each condition pair, one bar per model_type, mean +/- SEM cosine similarity."""
    fig, ax, _ = new_panel(ax, figsize=(6.2, 4.5))
    summaries = {}
    labels = None
    for mt in MODEL_TYPES:
        try:
            mean, sem, n_seeds, labels = CA.coding_angle_summary(mt, phase=phase)
        except FileNotFoundError:
            continue
        summaries[mt] = (mean, sem, n_seeds)
    if not summaries:
        raise FileNotFoundError(f"no coding-angle data for phase={phase}")

    n_stim = len(labels)
    pairs = [(i, j) for i in range(n_stim) for j in range(i + 1, n_stim)]
    pair_labels = [f"{labels[i]} vs {labels[j]}" for i, j in pairs]

    x = np.arange(len(pairs))
    width = 0.8 / max(len(summaries), 1)
    for mi, (mt, (mean, sem, n_seeds)) in enumerate(summaries.items()):
        vals = [mean[i, j] for i, j in pairs]
        errs = [sem[i, j] for i, j in pairs]
        color = FC.MODELS.get(mt, {}).get("color", f"C{mi}")
        label = FC.MODELS.get(mt, {}).get("label", mt)
        ax.bar(x + (mi - (len(summaries) - 1) / 2) * width, vals, width, yerr=errs, color=color, label=label, capsize=3)

    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(pair_labels)
    ax.set_ylabel("cosine similarity")
    ax.legend(frameon=False, fontsize=8)
    phase_label = "pre-reversal" if phase == "pre" else "post-reversal"
    ax.set_title(f"Coding-vector angle by model type ({phase_label})")
    return fig


def build_all(show_tag=None):
    for mt in MODEL_TYPES:
        try:
            fig = draw_coding_angle_heatmap_combined(mt)
        except FileNotFoundError as e:
            print(f"  (skip combined coding-angle heatmap {mt}: {e})")
            continue
        save_panel(fig, "Mechanistic/CodingAngle", f"MECH.ANGLE.combined.{mt}",
                   f"{mt}_coding_angle_heatmap_combined", show_tag)

    for phase in ("pre", "post"):
        for mt in MODEL_TYPES:
            try:
                fig = draw_coding_angle_heatmap(mt, phase=phase)
            except FileNotFoundError as e:
                print(f"  (skip coding-angle heatmap {mt}/{phase}: {e})")
                continue
            save_panel(fig, "Mechanistic/CodingAngle", f"MECH.ANGLE.{phase}.{mt}",
                       f"{mt}_coding_angle_heatmap_{phase}", show_tag)

        try:
            fig = draw_coding_angle_by_model(phase=phase)
        except FileNotFoundError as e:
            print(f"  (skip coding-angle by-model bars [{phase}]: {e})")
            continue
        save_panel(fig, "Mechanistic/CodingAngle", f"MECH.ANGLE.bymodel.{phase}",
                   f"coding_angle_by_model_{phase}", show_tag)


if __name__ == "__main__":
    build_all()
