"""Shared figure style for context-value-RNNs model figures.

Ported from the experimental analysis pipeline at
~/Documents/neuronal-representations/results/transfer, specifically:
  - figures/panels/transfer.mplstyle   -> model_figure_style.mplstyle (this dir)
  - figures/sat_plot_colours.py        -> STIM_COLOURS / lighten_hex below
  - figures/_common.py:condition_style -> condition_style below

Goal: new/updated model figures use these so they are visually consistent with
the real-data figures in results/transfer they are being compared against.
Existing figures.py / reversal_analysis.py in transfer_final/ and
reversal_study/ do NOT import this yet -- they use their own
code/figure_style.mplstyle (larger fonts, PDF+PNG, no tag stamp). Wiring this
in is a deliberate follow-up, not done automatically here, since it changes
already-reviewed output.
"""
from pathlib import Path
import matplotlib.pyplot as plt

STYLE_PATH = Path(__file__).parent / "model_figure_style.mplstyle"


def use_style():
    """Apply the ported experimental style globally (plt.style.use)."""
    plt.style.use(str(STYLE_PATH))


# Stimulus colours, ported verbatim (by hex value) from
# neuronal-representations/results/transfer/figures/sat_plot_colours.py
# STIM_STEM_COLOURS. That dict is keyed by reversal transition name
# ("100_to_0", "50", "0_to_100"); here it's re-keyed by the plain stimulus
# identity used throughout transfer_final/reversal_study ("0", "50", "100").
# FIXED (was backwards): identity-anchored stim "0" is the 0%-pre cue whose
# reward journey through the reversal is 0%->100% -- that's the real repo's
# "0_to_100" bucket (peach), not "100_to_0" (purple). Likewise stim "100" is
# the 100%-pre cue whose journey is 100%->0% -- "100_to_0" (purple), not
# "0_to_100". The previous version had these two swapped (a genuine bug,
# not a style choice -- caught while matching trace colours to the real
# figures on request; every panel using STIM_COLOURS before this fix had
# stim 0% and 100% coloured with each other's real-repo colour).
STIM_COLOURS = {
    "0": "#edb081",    # peach   (== "0_to_100" in the experimental repo: 0%->100%)
    "50": "#c24167",   # pink/magenta
    "100": "#4b2362",  # purple  (== "100_to_0" in the experimental repo: 100%->0%)
}

NONRESPONSIVE_COLOUR = "0.7"  # neutral grey, matches subgroups.py


def lighten_hex(hex_color, amount=0.55):
    """Blend a hex color toward white by `amount` (0 = no change, 1 = white).
    Ported from neuronal-representations/results/transfer/figures/_common.py.
    """
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = (int(c + (255 - c) * amount) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def condition_style(stim, phase="pre"):
    """Plot kwargs for one stimulus x phase, matching the experimental repo's
    pre=solid/post=dashed+lightened convention
    (neuronal-representations/results/transfer/figures/_common.py:condition_style).
    `stim` is "0" | "50" | "100"; `phase` is "pre" | "post".
    """
    color = STIM_COLOURS[str(stim)]
    if phase == "post":
        return dict(color=lighten_hex(color, 0.45), linestyle="--", lw=2.0)
    return dict(color=color, linestyle="-", lw=2.0)


# --- responder-group colours, ported from
# neuronal-representations/results/transfer/figures/panels/subgroups.py
# (GROUP_COL / _blend / _fine_color) so group-size bars use the same scheme:
# pure groups = their stimulus colour, pairwise = a 50/50 blend of the two
# stimulus colours, all-three = dark grey, non-responsive = light grey.
import numpy as np
import matplotlib.colors as _mcolors


def blend_hex(c1, c2):
    """50/50 RGB blend of two hex colours. Ported from subgroups.py's _blend()."""
    a = np.array(_mcolors.to_rgb(c1))
    b = np.array(_mcolors.to_rgb(c2))
    return tuple((a + b) / 2.0)


GROUP_COLOURS = {
    "0%-only": STIM_COLOURS["0"],
    "50%-only": STIM_COLOURS["50"],
    "100%-only": STIM_COLOURS["100"],
    "0% & 50%": blend_hex(STIM_COLOURS["0"], STIM_COLOURS["50"]),
    "0% & 100%": blend_hex(STIM_COLOURS["0"], STIM_COLOURS["100"]),
    "50% & 100%": blend_hex(STIM_COLOURS["50"], STIM_COLOURS["100"]),
    "all three": "0.25",
    "non-responsive": "0.7",
}
