"""circuit_diagram.py

Effective circuit diagram: functionally-defined unit subgroups as nodes (size ~
group membership), with SEPARATE excitatory and inhibitory edges between every
group pair -- including the same pair in both directions and self-connections
-- since this model has no Dale's law constraint, so a group-to-group
projection can (and does) carry both a net-positive and a net-negative
component simultaneously. Edge width ~ total |weight| summed over that
group-pair-sign's individual unit-to-unit connections (mean over seeds).

Input and readout-head nodes (each drawn as a circle) are each broken out
individually (not aggregated): the 3 specific OHE input dimensions actually
shown (stim 0%→100% / stim 50% / stim 100%→0%, from
circuit_graph.INPUT_LABELS) fan out above the group circle -- the "context"
input dim is dropped entirely (not just relabelled): for these single-context
runs it's a constant always-on bit carrying zero trial-to-trial information,
so drawing it just adds a node that never does anything informative. The
"response cue" input is also dropped (by request, for visual decluttering --
it's a real, informative input, see the chat's explanation of why it exists;
the full 5-dim picture is still available via weight_matrix_panel.py) --
and the specific readout heads actually present
for this model_type (actor/vigour_head, critic/value_head, and -- only for
classif_rl and classif_rl_readout_only -- the SSL stim_head; see
checkpoint_weights.HEAD_DISPLAY_NAMES) fan out below it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "style"))
sys.path.insert(0, str(_HERE.parent / "transfer" / "code"))
sys.path.insert(0, str(_HERE.parent / "analysis"))

from _tags import new_panel, save_panel  # noqa: E402
import style as S  # noqa: E402
import figure_config as FC  # noqa: E402
import circuit_graph as CG  # noqa: E402

MODEL_TYPES = CG.CW.MODEL_TYPES

EXCITATORY_COLOR = "#c0392b"
INHIBITORY_COLOR = "#2c5aa0"

NODE_COLOURS = {
    "broad": [S.NONRESPONSIVE_COLOUR, S.STIM_COLOURS["0"], S.STIM_COLOURS["50"], S.STIM_COLOURS["100"]],
    "fine": [
        S.GROUP_COLOURS["non-responsive"],
        S.GROUP_COLOURS["0%-only"],
        S.GROUP_COLOURS["50%-only"],
        S.GROUP_COLOURS["100%-only"],
        S.GROUP_COLOURS["0% & 50%"],
        S.GROUP_COLOURS["0% & 100%"],
        S.GROUP_COLOURS["50% & 100%"],
        S.GROUP_COLOURS["all three"],
    ],
}

# curvature ("rad" in matplotlib's arc3 connection style) for each of the 4 possible
# directed/signed edges between an ordered pair (s -> t), so they never overlap
_RAD = {"pos": 0.18, "neg": 0.36}
MIN_EDGE_FRAC = 0.03  # don't draw an edge weaker than this fraction of the strongest edge in the panel


def _node_positions(n_groups, radius=1.0):
    angles = np.linspace(0, 2 * np.pi, n_groups, endpoint=False) + np.pi / 2
    return np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)


def _fan_positions(n, y, half_width=1.35):
    """n node positions spread evenly along x at fixed height y (used for the
    specific-input-dimension row above the group circle, and the specific-
    readout-head row below it)."""
    if n <= 1:
        xs = np.zeros(1)
    else:
        xs = np.linspace(-half_width, half_width, n)
    return np.stack([xs, np.full(n, y)], axis=1)


def _draw_self_loop(ax, xy, node_r, sign, magnitude, max_mag, colour, angle_offset):
    if magnitude <= 0 or max_mag <= 0:
        return
    frac = magnitude / max_mag
    if frac < MIN_EDGE_FRAC:
        return
    loop_r = node_r * (0.35 + 0.55 * frac)
    lw = 0.6 + 3.5 * frac
    cx, cy = xy
    # push the loop outward along the node's own radial direction, offset left/right for exc vs inh
    ang = np.arctan2(cy, cx) + angle_offset
    off = node_r * 1.15
    loop_xy = (cx + off * np.cos(ang), cy + off * np.sin(ang))
    ax.add_patch(Circle(loop_xy, loop_r, fill=False, edgecolor=colour, linewidth=lw, alpha=0.85, zorder=2))


def _draw_edge(ax, xy_s, xy_t, node_r, sign, magnitude, max_mag, colour, rad):
    if magnitude <= 0 or max_mag <= 0:
        return
    frac = magnitude / max_mag
    if frac < MIN_EDGE_FRAC:
        return
    lw = 0.6 + 5.5 * frac
    patch = FancyArrowPatch(
        xy_s, xy_t,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>", mutation_scale=8 + 10 * frac,
        shrinkA=node_r * 72, shrinkB=node_r * 72,
        color=colour, linewidth=lw, alpha=0.55 + 0.4 * frac, zorder=1,
    )
    ax.add_patch(patch)


def draw_circuit_diagram(model_type, granularity="broad", phase="pre", ax=None, seeds=None, show_io=True,
                         omit_silent=True):
    fig, ax, owns = new_panel(ax, figsize=(6.8, 7.8) if show_io else (6.2, 6.2))
    n_per_group, pos_mat, neg_mat, n_seeds = CG.group_edges_averaged(model_type, granularity=granularity, seeds=seeds, phase=phase)
    labels_full = CG.GRANULARITIES[granularity]["labels"]
    colours_full = NODE_COLOURS[granularity]

    # Both granularities' first label is "silent" (broad) / "non-responsive" (fine)
    # -- the catch-all bin for units that showed no stimulus-evoked tuning at all.
    # Dropped by default (omit_silent=True): it's rarely the interesting part of
    # an "effective circuit" and its edges tend to dominate the plot's dynamic
    # range, crowding out the functionally meaningful groups.
    if omit_silent:
        keep = [lbl not in ("silent", "non-responsive") for lbl in labels_full]
    else:
        keep = [True] * len(labels_full)
    keep_idx = [i for i, k in enumerate(keep) if k]

    labels = [labels_full[i] for i in keep_idx]
    colours = [colours_full[i] for i in keep_idx]
    n_per_group = n_per_group[keep_idx]
    pos_mat = pos_mat[np.ix_(keep_idx, keep_idx)]
    neg_mat = neg_mat[np.ix_(keep_idx, keep_idx)]
    n_groups = len(labels)

    # Functional-group labels are stim-INDEX-based (which physical stimulus a
    # group prefers), not reward-value-based, but BROAD_LABELS' text ("pref 0%"
    # etc.) only reads correctly pre-reversal -- physical stim index 0 delivers
    # 0% reward pre-reversal but 100% post-reversal (VM_PRE=[0,.5,1] ->
    # VM_POST=[1,.5,0], 50% stim unchanged; see circuit_graph.INPUT_LABELS'
    # comment for the full explanation). Relabel with the same transition
    # notation used there so a "pref 0%" group's node reads "pref 0%→100%"
    # in BOTH the pre- and post-reversal diagram -- same label either way,
    # naming the physical group's full reward trajectory across the reversal,
    # rather than a bare "0%"/"100%" that would silently mean opposite things
    # depending on which diagram it's in.
    GROUP_LABEL_OVERRIDE = {"pref 0%": "pref 0%\u2192100%", "pref 100%": "pref 100%\u21920%"}
    display_labels = [GROUP_LABEL_OVERRIDE.get(l, l) for l in labels]

    xy = _node_positions(n_groups, radius=1.0)
    frac_units = n_per_group / n_per_group.sum()
    node_r = 0.10 + 0.16 * np.sqrt(frac_units / frac_units.max())

    max_mag = max(pos_mat.max(), neg_mat.max(), 1e-12)

    if show_io:
        in_pos, in_neg, out_pos, out_neg, head_names, _io_n_seeds = CG.group_input_readout_split_averaged(
            model_type, granularity=granularity, seeds=seeds, phase=phase)
        in_pos = in_pos[keep_idx]; in_neg = in_neg[keep_idx]
        out_pos = out_pos[keep_idx]; out_neg = out_neg[keep_idx]

        # Omit the "context" input dim entirely (not just relabel it): see
        # module docstring above. Same keep_idx-style filtering pattern as
        # the silent-group omission, applied to the input-dimension axis.
        input_labels_full = CG.INPUT_LABELS
        # Omit "context" (constant, see above) AND "response cue" -- also
        # requested dropped for visual clarity; the full 5-dim picture
        # (including the cue) is still available via weight_matrix_panel.py's
        # figures for anyone who wants it.
        OMIT_INPUT_PREFIXES = ("context", "response cue")
        input_keep_idx = [i for i, lbl in enumerate(input_labels_full) if not lbl.startswith(OMIT_INPUT_PREFIXES)]
        input_labels = [input_labels_full[i] for i in input_keep_idx]
        in_pos = in_pos[:, input_keep_idx]
        in_neg = in_neg[:, input_keep_idx]

        n_inputs = in_pos.shape[1]
        n_heads = out_pos.shape[1]
        io_max = max(in_pos.max(), in_neg.max(),
                     out_pos.max() if n_heads else 0.0,
                     out_neg.max() if n_heads else 0.0, 1e-12)
        input_xy = _fan_positions(n_inputs, y=1.95)
        readout_xy = _fan_positions(max(n_heads, 1), y=-1.95)
        io_r = 0.11

    # off-diagonal edges (both directions, both signs)
    for s in range(n_groups):
        for t in range(n_groups):
            if s == t:
                continue
            _draw_edge(ax, xy[s], xy[t], node_r[t], "pos", pos_mat[t, s], max_mag, EXCITATORY_COLOR, _RAD["pos"])
            _draw_edge(ax, xy[s], xy[t], node_r[t], "neg", neg_mat[t, s], max_mag, INHIBITORY_COLOR, -_RAD["neg"])

    # self-connections
    for g in range(n_groups):
        _draw_self_loop(ax, xy[g], node_r[g], "pos", pos_mat[g, g], max_mag, EXCITATORY_COLOR, angle_offset=-0.35)
        _draw_self_loop(ax, xy[g], node_r[g], "neg", neg_mat[g, g], max_mag, INHIBITORY_COLOR, angle_offset=0.35)

    if show_io:
        # specific input dims -> every group
        for i in range(n_inputs):
            for g in range(n_groups):
                _draw_edge(ax, input_xy[i], xy[g], node_r[g], "pos", in_pos[g, i], io_max, EXCITATORY_COLOR, 0.10)
                _draw_edge(ax, input_xy[i], xy[g], node_r[g], "neg", in_neg[g, i], io_max, INHIBITORY_COLOR, -0.20)
        # every group -> specific readout heads
        for h in range(n_heads):
            for g in range(n_groups):
                _draw_edge(ax, xy[g], readout_xy[h], io_r, "pos", out_pos[g, h], io_max, EXCITATORY_COLOR, 0.10)
                _draw_edge(ax, xy[g], readout_xy[h], io_r, "neg", out_neg[g, h], io_max, INHIBITORY_COLOR, -0.20)

        for i in range(n_inputs):
            ax.add_patch(Circle(input_xy[i], io_r * 1.15, facecolor="0.85", edgecolor="0.2",
                                linewidth=1.1, zorder=3))
            ax.text(input_xy[i, 0], input_xy[i, 1] + io_r * 1.4 + 0.13, input_labels[i],
                    ha="center", va="center", fontsize=7)
        if n_heads:
            for h in range(n_heads):
                ax.add_patch(Circle(readout_xy[h], io_r * 1.15, facecolor="0.85", edgecolor="0.2",
                                    linewidth=1.1, zorder=3))
                ax.text(readout_xy[h, 0], readout_xy[h, 1] - io_r * 1.4 - 0.13, head_names[h],
                        ha="center", va="center", fontsize=7)
        else:
            ax.text(readout_xy[0, 0], readout_xy[0, 1], "(no readout heads found)",
                    ha="center", va="center", fontsize=7, color="0.4")

    for g in range(n_groups):
        ax.add_patch(Circle(xy[g], node_r[g], facecolor=colours[g], edgecolor="0.2", linewidth=1.2, zorder=3))
        label_r = 1.0 + node_r[g] + 0.16
        ang = np.arctan2(xy[g, 1], xy[g, 0])
        lx, ly = label_r * np.cos(ang), label_r * np.sin(ang)
        ax.text(lx, ly, f"{display_labels[g]}\n(n={n_per_group[g]:.1f} avg)", ha="center", va="center", fontsize=8)

    ax.set_xlim(-2.15, 2.15)
    ax.set_ylim((-2.5, 2.5) if show_io else (-1.7, 1.7))
    ax.set_aspect("equal")
    ax.axis("off")

    mdl_label = FC.MODELS.get(model_type, {}).get("label", model_type)
    phase_label = "pre-reversal" if phase == "pre" else "post-reversal"
    ax.plot([], [], color=EXCITATORY_COLOR, lw=2.5, label="excitatory (sum + weights)")
    ax.plot([], [], color=INHIBITORY_COLOR, lw=2.5, label="inhibitory (sum |- weights|)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.06), frameon=False, fontsize=8, ncol=2)
    ax.set_title(f"{mdl_label}: {granularity} functional circuit, {phase_label}\n"
                 f"(mean edge weight \u00b1 group membership across n={n_seeds} seeds -- not one example network)")
    return fig


def build_all(show_tag=None):
    for granularity in CG.GRANULARITIES:
        for phase in ("pre", "post"):
            for mt in MODEL_TYPES:
                try:
                    fig = draw_circuit_diagram(mt, granularity=granularity, phase=phase)
                except FileNotFoundError as e:
                    print(f"  (skip circuit diagram {mt}/{granularity}/{phase}: {e})")
                    continue
                save_panel(fig, "Mechanistic/CircuitDiagram", f"MECH.CIRCUIT.{granularity}.{phase}.{mt}",
                           f"{mt}_circuit_diagram_{granularity}_{phase}", show_tag)


if __name__ == "__main__":
    build_all()
