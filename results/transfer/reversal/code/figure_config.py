"""Editable figure config — colours and labels in ONE place.
(Font sizes / spines / dpi live in the matplotlib style sheet, figure_style.mplstyle.)

Imported by figures.py. Edit the hex codes / labels here and re-run to restyle.
"""

# Per-stimulus colours, used by the trial-aligned PSTH group grid.
STIM_COLORS = {
    0: "#fab079",   # 0%
    1: "#b8397a",   # 50%
    2: "#2E215E",   # 100%
}

# Per-stimulus display labels (order matches STIM_COLORS keys 0,1,2).
STIM_LABELS = ["0%", "50%", "100%"]

# Per-model colours + labels, used by the bar plots (colours differ by model).
MODELS = {
    "classif_rl":              {"label": "SSL + RL",                     "color": "#aa3377"},
    "rl_only":                 {"label": "RL only",                      "color": "#3377bb"},
    "classif_rl_readout_only": {"label": "SSL + RL (readout-only RL)",   "color": "#229977"},
}
