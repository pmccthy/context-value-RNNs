"""weight_matrix_panel.py

The trained RNN's weight matrices themselves (not an aggregate summary):
one big square heatmap of the recurrent (hidden->hidden) weight matrix, with
units re-sorted by their functional subgroup (same broad/fine categories as
circuit_diagram.py / model_group_categories.py) so within/between-group
block structure is visible, PLUS matching input->hidden and hidden->readout
strips using the SAME unit ordering along their unit axis.

Referenced in chat as "a plot like this we were making at some point" --
searched the repo and found no prior version of this specific figure (see
chat), so this is a fresh implementation, not a port.

NECESSARILY single-seed (not seed-averaged, unlike circuit_diagram.py's
group-aggregated edge sums): a raw unit x unit weight matrix has no
correspondence across independently-initialised/trained networks, so
"averaging" one would mix unrelated units. Each figure is explicitly titled
as one example seed. Defaults to the lowest-numbered recovered seed for
reproducibility.
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
import checkpoint_weights as CW  # noqa: E402
import circuit_graph as CG  # noqa: E402
import vigour_value as VV  # noqa: E402

MODEL_TYPES = CW.MODEL_TYPES

GROUP_LABEL_OVERRIDE = {"pref 0%": "pref 0%→100%", "pref 100%": "pref 100%→0%"}


def _output_specs(sd):
    specs = []
    for hk in CW.HEAD_KEYS:
        if hk not in sd:
            continue
        out_dim = sd[hk].shape[0]
        if out_dim == 1:
            specs.append((hk, 0, CW.HEAD_SHORT_NAMES.get(hk, hk)))
        else:
            for ri in range(out_dim):
                label = CG.STIM_HEAD_ROW_LABELS[ri] if ri < len(CG.STIM_HEAD_ROW_LABELS) else f"{hk}[{ri}]"
                specs.append((hk, ri, label))
    return specs


def draw_weight_matrix_by_subgroup(model_type, phase="pre", seed=None, granularity="broad", fig=None,
                                   subplot_spec=None):
    """One example network's weight matrices, unit-sorted by functional
    subgroup: big square h2h matrix in the centre, input->hidden strip above
    it, hidden->readout strip to its right -- all three sharing the SAME
    sorted unit ordering along their shared axis, with thin lines marking
    subgroup boundaries.
    """
    if seed is None:
        rec = VV.recovered_seeds(model_type)
        seed = min(rec) if rec else CW.SEEDS[0]

    cats, n_groups = CG._cats_for(model_type, seed, granularity, phase)
    labels_full = CG.GRANULARITIES[granularity]["labels"]
    order = np.argsort(cats, kind="stable")
    group_sizes = [int((cats == g).sum()) for g in range(n_groups)]
    boundaries = np.cumsum(group_sizes)[:-1]
    centres = np.cumsum(group_sizes) - np.asarray(group_sizes) / 2.0
    disp_labels = [GROUP_LABEL_OVERRIDE.get(labels_full[g], labels_full[g]) for g in range(n_groups)]

    sd = CG._final_state(model_type, seed, phase)
    Wh = sd[CW.RECURRENT_KEY][np.ix_(order, order)]           # (128,128) target x source, sorted both axes
    Win = sd[CW.INPUT_KEY][order, :].T                        # (n_input, 128) sorted along unit (col) axis

    specs = _output_specs(sd)
    n_hidden = sd[CW.RECURRENT_KEY].shape[0]
    Wout = np.full((n_hidden, len(specs)), np.nan)
    for hi, (hk, ri, _lbl) in enumerate(specs):
        row = sd[hk][ri, :]
        Wout[: row.shape[0], hi] = row
    Wout = Wout[order, :]                                      # sorted along unit (row) axis
    head_labels = [lbl for _, _, lbl in specs]

    own_fig = fig is None
    if own_fig:
        fig = plt.figure(figsize=(8.5, 8.5))
    outer = (subplot_spec.subgridspec(2, 2, width_ratios=[7, 1.6], height_ratios=[1.6, 7], hspace=0.06, wspace=0.06)
             if subplot_spec is not None else
             fig.add_gridspec(2, 2, width_ratios=[7, 1.6], height_ratios=[1.6, 7],
                              hspace=0.06, wspace=0.06, left=0.12, right=0.97, top=0.90, bottom=0.10))

    ax_in = fig.add_subplot(outer[0, 0])
    ax_main = fig.add_subplot(outer[1, 0])
    ax_out = fig.add_subplot(outer[1, 1])

    vmax_h = np.nanpercentile(np.abs(Wh), 99) or 1.0
    ax_main.imshow(Wh, cmap="RdBu_r", vmin=-vmax_h, vmax=vmax_h, aspect="auto", interpolation="nearest")
    for b in boundaries:
        ax_main.axhline(b - 0.5, color="k", lw=0.7)
        ax_main.axvline(b - 0.5, color="k", lw=0.7)
    ax_main.set_xticks(centres); ax_main.set_xticklabels(disp_labels, rotation=45, ha="right", fontsize=9)
    ax_main.set_yticks(centres); ax_main.set_yticklabels(disp_labels, fontsize=9)
    ax_main.set_xlabel("source unit (sorted by group)", fontsize=14)
    ax_main.set_ylabel("target unit (sorted by group)", fontsize=14)

    vmax_in = np.nanpercentile(np.abs(Win), 99) or 1.0
    ax_in.imshow(Win, cmap="RdBu_r", vmin=-vmax_in, vmax=vmax_in, aspect="auto", interpolation="nearest")
    for b in boundaries:
        ax_in.axvline(b - 0.5, color="k", lw=0.6)
    ax_in.set_xticks([])
    ax_in.set_yticks(range(len(CG.INPUT_LABELS))); ax_in.set_yticklabels(CG.INPUT_LABELS, fontsize=7.5)
    ax_in.set_title(f"{FC.MODELS.get(model_type, {}).get('label', model_type)}: weight matrices, "
                    f"{'pre' if phase == 'pre' else 'post'}-reversal (example seed{seed})", fontsize=10)

    vmax_out = np.nanpercentile(np.abs(Wout[~np.isnan(Wout)]), 99) if np.isfinite(Wout).any() else 1.0
    vmax_out = vmax_out or 1.0
    ax_out.imshow(Wout, cmap="RdBu_r", vmin=-vmax_out, vmax=vmax_out, aspect="auto", interpolation="nearest")
    for b in boundaries:
        ax_out.axhline(b - 0.5, color="k", lw=0.6)
    ax_out.set_yticks([])
    ax_out.set_xticks(range(len(head_labels))); ax_out.set_xticklabels(head_labels, rotation=90, fontsize=7.5)

    return fig


def build_all(show_tag=None):
    for phase in ("pre", "post"):
        for mt in MODEL_TYPES:
            try:
                fig = draw_weight_matrix_by_subgroup(mt, phase=phase)
            except FileNotFoundError as e:
                print(f"  (skip weight matrix {mt}/{phase}: {e})")
                continue
            save_panel(fig, "Mechanistic/WeightMatrix", f"MECH.WMAT.{phase}.{mt}",
                       f"{mt}_weight_matrix_{phase}", show_tag)


if __name__ == "__main__":
    build_all()
