# unified_figures/

A single place to generate every figure from the context-value-RNNs modelling
work, without moving or duplicating the underlying data. Built **in place**
inside this repo (not yet merged into `~/Documents/transfer` — modelling
results, especially `reversal_study/`, are still changing).

Everything under `transfer/` and `reversal/` is **symlinks** into the existing
`transfer_final/`, `reversal_study/`, and repo-root `scripts/` folders — no
data was copied or moved, so this folder stays in sync with those
automatically and nothing here is a second source of truth. The only new
content is `style/` (ported figure style, see below) and
`cross_model_vs_experiment/` (new analyses combining model + real experimental
data).

```
unified_figures/
├── FIGURE_LIST.md              <- the master figure list / status doc, start here
├── style/                      <- NEW: shared style ported from the experimental figures
│   ├── model_figure_style.mplstyle
│   └── style.py
├── transfer/                   <- 3-model comparison (symlinks -> transfer_final/)
│   ├── code, jobs (=reproducibility/), model_runs*, figure_data*, output*
├── reversal/                   <- reversal-learning study (symlinks -> reversal_study/ + scripts/)
│   ├── code, data (=results/)
│   └── jobs/                   <- train_model.py / train_reversal.py / run_inference.py (from
│                                   repo-root scripts/, since reversal_study/reproducibility/training
│                                   is currently empty -- see "Known issues" in FIGURE_LIST.md) plus
│                                   the five run_*_full.sh / run_reward_scale_*.sh study launchers
└── cross_model_vs_experiment/  <- NEW: chi-squared model-fit + planned pipeline-parity analyses
```

Environments: figure-generation scripts here only need numpy/pandas/matplotlib/scipy
(no torch) once `figure_data.pkl` exists — see each script's own `--help`.
Training/inference (`jobs/train_model.py`, `train_reversal.py`, `run_inference.py`,
and the `run_*_full.sh` sweep launchers) need the `cxval` env; the experimental-side
extraction in `cross_model_vs_experiment/` needs the `neuronal_representations` env
(really just pandas).
