# Final 3s1c continuous-vigour models — transfer bundle

Self-contained bundle of the trained models, the data extracted from them, the code
that produces every figure, and a source snapshot for reproducibility. Given the
trained models in `model_runs/`, running the inference step regenerates all figure
data, and the figure code regenerates every figure (PNG + PDF).

## The task and the three models

A continuous-action (lick *vigour*) RNN performs the 3-stimulus / 1-context task:
three stimuli with reward probability **0% / 50% / 100%**, randomly interleaved.
Three model types are compared (identical except for how the hidden units are shaped):

- **`classif_rl`** — RL **+** an auxiliary stimulus-classification head (supervised in
  the reward window). This is the model that yields distinct groups of neurons each
  responding *exclusively* to one stimulus.
- **`rl_only`** — matched RL-only baseline (shared value-magnitude code; the low/mid
  exclusive groups collapse).
- **`classif_rl_readout_only`** — `classif_rl`, but the RL actor/critic gradients are
  **stopped at the readout** (`detach_readout=True`): the hidden units are shaped
  *only* by the classification (+ activity) losses, and the vigour/value heads are
  linear readouts trained off — but not into — that representation.

All arms use an ITI-targeted L2 firing-rate penalty (`activity_coef=2.0`,
`activity_at=iti`) that keeps the pre-stimulus baseline near the ReLU floor.

> The `classif_rl` and `rl_only` arms have **30 seeds** (42–71). The
> `classif_rl_readout_only` arm here is a **2-seed smoke set**; train the full 30 with
> `reproducibility/training/run_all.sh`.

## Layout

```
transfer_final/
  README.md  requirements.txt  environment.yml
  model_runs/                      trained models, grouped by model type
    classif_rl/seed<NN>/           model.pt, model_init.pt, config.json, meta.txt
    rl_only/seed<NN>/
    classif_rl_readout_only/seed<NN>/
  figure_data/                     everything the figures read
    figure_data.pkl                self-documenting dict (dims + coords on every array)
    metrics.csv                    tidy scalar metrics (model_type, seed, stim)
    activations.npz                per-unit stim tuning (3, H) keyed "{type}_{seed}"
    time_resolved/                 raw per-trial activity, one .npz per model
  figures/                         all output figures (PNG + PDF)
  code/                            generate / explore (reads figure_data only)
    figures.py                     figure library + script
    figure_style.mplstyle          editable matplotlib style sheet
    make_figures.ipynb             one cell per figure, editable
    explore_figure_data.ipynb      structure of figure_data.pkl
    explore_time_resolved.ipynb    structure of the per-trial .npz files
  reproducibility/                 regenerate the data from scratch
    cxval/                         source snapshot (the importable package)
    training/  train_model.py, run_all.sh
    inference/ run_inference.py
    requirements.txt
```

## Generate the figures (from the stored data — no PyTorch needed)

```bash
pip install -r requirements.txt          # numpy, pandas, matplotlib, jupyterlab
python code/figures.py --data figure_data --out figures
# or open code/make_figures.ipynb and run it (one cell per figure)
```

Style is controlled by `code/figure_style.mplstyle` (fonts, spines, sizes, dpi); edit
it and re-run. Colours/labels per model live in `figures.MODELS`. The figure code is
built from small reusable primitives that take *which data* and *which models* to plot:

- `bar_metric(D, metric, model_types=...)` — grouped bars per stimulus, coloured by model.
- `heatmap_seed(D, model_type, seed)` — one model+seed per-unit tuning heatmap.
- `heatmap_montage(D, model_type)` — those heatmaps tiled across a model's seeds.
- `group_grid(D, model_type)` — the responsiveness-group trial-aligned (PSTH) grid.

Thin wrappers (`fig_bar_metric`, `fig_heatmap_montage`, `fig_group_grid`) save PNG+PDF;
`make_all(D, out)` writes the lot.

### Figures produced (each as `.png` + `.pdf`)

- `responsive_per_stim[ _<model> ]` — # significantly responsive units per stimulus.
- `population_activity_per_stim[ _<model> ]` — mean hidden activation per stimulus.
- `vigour_per_stim[ _<model> ]` — lick vigour per stimulus.
  (The three scalar metrics come combined and once per model; the combined/per-model
  versions of a metric **share one y-axis** for comparability.)
- `activation_heatmaps_<model>` — per-unit stim-window activation, one panel per seed.
- `celltype_group_grid_<model>` — trial-aligned mean response of each
  responsiveness-defined group (exclusives on the diagonal, pairs below, all-three
  corner), pooled over seeds, for each presented stimulus.

## The data files

`figure_data.pkl` is the source of truth and self-documents via `dims`/`coords`:

- `scalars[<metric>]` — `(model_type, seed, stim)` for `vigour`, `pop_activity`,
  `n_responsive`, `frac_responsive` (`NaN` where a (type, seed) is absent).
- `tuning.data` — `(model_type, seed, stim, unit)` per-unit stim-window mean.
- `aligned_mean.data` — `(model_type, seed, unit, time, stim)` per-unit trial-aligned
  mean; the `time` axis is the `ITI | stim | outcome` window (`period.segment_bounds`).
- `responsive.data` — `(model_type, seed, unit, stim)` bool; defines the cell-type groups.

`metrics.csv` is the scalar block in tidy long form. `activations.npz` is the per-unit
tuning as flat `(3, H)` arrays. `time_resolved/<type>_seed<NN>.npz` holds the **raw
per-trial** activity: `aligned` `(trial, time, unit)`, `stimulus`, `vigour`, `bounds`.
See the two `explore_*` notebooks.

### Where the metrics come from / rebuilding them without the repo

The bar-plot metrics (vigour, population activity, responder counts/fractions) are
originally computed by `reproducibility/inference/run_inference.py` during the model
rollout — that needs PyTorch + `cxval`. But they are all derivable from the stored
`time_resolved/*.npz` files, so you can rebuild `figure_data.pkl` + `metrics.csv` +
`activations.npz` **without any model or inference** (numpy + scipy only):

```bash
python code/build_figure_data_from_timeresolved.py \
       --time-resolved figure_data/time_resolved --out figure_data
```

This reproduces `run_inference`'s numbers exactly (same arrays, same paired t-test),
then `code/figures.py` regenerates the figures as usual. The responder rule is selectable
with `--direction {excitatory,two_sided,suppressed}` (default `excitatory` = significantly
*above* baseline, as used everywhere else).

`code/compare_responder_definitions.py` uses this to compare responder **definitions**
side by side — **positive-only** (excitatory) vs **two-sided** (any significant
difference, incl. suppressed units) — saving both bar plots (responders per stimulus, and
the responder groups) under each. It shows that the exclusive/graded structure is a
property of the positive-only rule (two-sided collapses almost everything into "all three").
The same comparison is in the appendix of `make_figures.ipynb`.

## Inference settings

Each model is evaluated deterministically on **1,600 interleaved trials**
(`8 episodes × 200 trials`, ≈530 per stimulus; the knob is at the top of
`reproducibility/inference/run_inference.py`). The 3 stimuli are randomly shuffled
within every episode. All metrics, tuning, responsiveness and the trial-aligned
activity come from this single pass.

## Reproduce the data

See `reproducibility/README.md`. In short, from the bundle root:

```bash
pip install -r reproducibility/requirements.txt
bash reproducibility/training/run_all.sh                 # train all arms × seeds
python reproducibility/inference/run_inference.py \
       --runs model_runs --out figure_data               # rebuild figure_data + time_resolved
python code/figures.py --data figure_data --out figures  # rebuild figures
```

## Reversal experiment (contingency reversal)

Continue-trains every trained model on a **reversal** — the 0% and 100% reward
contingencies are swapped (`value_matrix [0,.5,1] → [1,.5,0]`; the 50% stimulus and all
stimulus *inputs* are unchanged) — warm-started from the trained weights (fresh optimizer,
auxiliary classification loss kept on, since identity is unchanged). It asks (a) does each
class recover pre-reversal performance, and (b) how does its representation reorganise.

```bash
bash reproducibility/training/run_reversal_all.sh                    # -> model_runs_reversal/
python reproducibility/inference/run_inference.py \
       --runs model_runs_reversal --out figure_data_reversal         # reversed figure data
python code/figures.py --data figure_data_reversal --out figures_reversal   # standard figures, reversed
python code/reversal_analysis.py --pre figure_data --post figure_data_reversal \
       --reversal-runs model_runs_reversal --out figures_reversal
```

`reversal_analysis.py` produces:
- **`reversal_recovery`** — reward on the reversed task vs update, per class, with each
  class's pre-reversal performance as a dotted line (does it climb back?).
- **`reversal_pref_transition`** — per class, a 3×3 matrix of how units' preferred stimulus
  maps pre→post (over pre-reversal responders): **diagonal** = code kept its stimulus
  *identity* tuning (value re-read in the readout), **anti-diagonal** = code followed the
  value swap.
- **`reversal_tuning_correlation`** — per class, mean per-unit correlation of pre vs post
  tuning (1 = unchanged, <0 = flipped): a one-number summary of the above.
- **`reversal_bars_{vigour,activity,responders}`** — per class, pre (grey) vs post (colour)
  bars per stimulus at end of training.
- **`reversal_timeline_{vigour,activity,selectivity}`** — each metric per stimulus on one
  **trials axis spanning original training → reversal**, with the reversal point dashed
  (e.g. 0% vigour rises and 100% falls after the reversal). Driven by periodic probes logged
  during training (`--probe-every`, stored in each run's `history.json` as `probe_*`);
  `run_all.sh` and `run_reversal_all.sh` probe by default. (If the original runs weren't
  probed, this falls back to a reversal-phase-only timeline with the pre level dotted.)
- **`reversal_heatmaps_prepost`** — per class, pre vs post per-unit tuning heatmap (units
  sorted by pre-reversal preferred), so the reorganisation is directly visible.

Each reversed run's `history.json` also stores the recovery curve and the pre/post reward
(`recovered_fraction`). Warm-starting the full actor-critic is via
`train_vigour(..., init_model=state_dict)`; training-time metric probes via
`train_vigour(..., probe_every=N, probe_fn=...)`.
