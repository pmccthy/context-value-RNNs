# Figure list — context-value-RNNs

Draft master list of figures from current modelling results, compiled
2026-08-27. Modelling (especially `reversal_study/`) is still in progress, so
treat status notes below as a snapshot, not a final state. Organized by study;
each entry gives the figure, what it shows, where it's generated, and status.

Style: sections A-C use the existing model-figure style
(`code/figure_style.mplstyle`, PDF+PNG, larger fonts). New work in section D
uses the newly-ported experimental style (`unified_figures/style/`, PNG-only,
`transfer.mplstyle` conventions) -- see "Style unification" below for the plan
to bring A-C in line.

## A. Transfer study -- 3-model comparison

Source: `unified_figures/transfer/` (symlinks -> `transfer_final/`). Three
model types throughout: `rl_only`, `classif_rl` ("SSL + RL"),
`classif_rl_readout_only`. Generated via `transfer/code/figures.py`.

| figure | shows | status |
|---|---|---|
| `responsive_per_stim` (+ per-model) | count of significantly-responsive units per stimulus (0/50/100%), grouped bars per model | done, 30 seeds classif_rl & rl_only |
| `population_activity_per_stim` (+ per-model) | mean hidden-unit activation per stimulus per model | done |
| `vigour_per_stim` (+ per-model) | mean lick vigour per stimulus per model | done |
| `responder_groups` | unit counts per 7-way responder group (3 exclusive, 3 pairwise, all-three), grouped bars per model | done |
| `activation_heatmaps_<model>` | per-unit stimulus-tuning heatmap, tiled across all seeds | done |
| `heatmap_<model>_seed42` | same heatmap, single seed detail | done |
| `celltype_group_grid_<model>` | 3x3 PSTH grid, mean+-SEM trial-aligned activity per responder group per stimulus | done |
| `responder_count_per_stim_<def>` / `responder_groups_<def>` (positive vs. two_sided) | sensitivity check: exclusive-cell structure only holds under the excitatory-only responder definition | done (`compare_responder_definitions.py`) |

**Status caveat:** `classif_rl_readout_only` currently ships as a 2-seed smoke
set in `transfer_final/model_runs/`, not the full 30 -- its figures above are
not yet on equal footing with the other two model types. Re-running
`reproducibility/training/run_all.sh` for that model type is a prerequisite
before treating any 3-way comparison here as final.

## B. Reversal-learning study -- baseline (2500 & 5000 trial budgets)

Source: `unified_figures/reversal/` (symlinks -> `reversal_study/results/reversal_2500`,
`reversal_5000`). Same 3 model types, continue-trained through a 0%<->100%
reward-contingency swap.

| figure | shows | status |
|---|---|---|
| `reversal_recovery` | reward on reversed task vs. update, dotted = pre-reversal baseline | done, both budgets |
| `reversal_recovery_fraction` | same, normalized to %-of-own-baseline | done |
| `reversal_recovered_proportion` | % of seeds recovering above threshold, per model (`--min-recovery` runs) | done |
| `reversal_pref_transition` | 3x3 pre->post preferred-stimulus transition matrix per model | done |
| `reversal_tuning_correlation` | mean pre/post tuning-vector correlation per model | done |
| `reversal_bars_{vigour,activity,responders}` | pre (grey) vs. post (colored) bars per stimulus, per model | done |
| `reversal_timeline_{vigour,activity,selectivity}` | metric across training, spanning original training into reversal | done |
| `reversal_heatmaps_prepost` | pre vs. post per-unit tuning heatmap, one seed | done |
| `rsa_rdm_heatmaps(_stim)` | time-resolved / stimulus-only representational dissimilarity matrices | done |
| `rsa_second_order(_stim)` | model x model correlation-of-RDMs | done |
| seed-level suite (`seed_groups.py`): recovery strip, tuning-correlation & activity-diff violins (recovered vs. failed), responder-group proportions/activity seedwise, alluvial/Sankey diagrams, RDM-per-group, per-seed vigour/RPE/value curves | seed-by-seed breakdown of everything above, split by whether that seed recovered | done |
| `recovery_time`, `recovery_definition_comparison` | first-crossing time-to-recovery under 3 different definitions | done (`recovery_time.py`) |

## C. Mechanistic intervention studies

Each produces the same reversal-analysis + seed_groups + recovery_time figure
suite as section B, scoped to that intervention's own 30-seed run.

| study | manipulation | status |
|---|---|---|
| `action_std_0p15_full` | wider action noise (0.15 vs. baseline 0.05) | complete, 90/90 seeds both phases |
| `min_vigour_0p1_full` | vigour floor at 0.1 instead of 0 | complete, 90/90 |
| `squash_1p3_full` | smooth soft-clamp instead of hard [0,1] clamp | **incomplete** -- 81/90 original-training seeds trained, reversal phase not started; run looks stalled |
| `reward_scale_intervention` (control / boost_up / damp_down) | causal reward-scale manipulation on RPE during reversal only, 8-seed pilot | complete, `figures_comparison/` done |
| `reward_scale_dose_response` | 5-point dose-response on the same manipulation | **not run** -- only the 2 reused magnitudes (control, damp_down) exist; the 3 new scales (0.7, 0.5, 0.1) were never trained |
| `terminal_rpe`, `terminal_value_minvig` | synthetic terminal-RPE/value substitute for runs lacking checkpoints | complete |
| `action_std_pilot`, `vigour_floor_pilot` | 5-seed pilots preceding the two `_full` studies above | complete (superseded by the full runs) |

## D. New: cross-model-vs-experiment analyses

The core new ask -- comparing model results directly against the real
experimental figures in `~/Documents/neuronal-representations/results/transfer`.
See `cross_model_vs_experiment/README.md` for full detail.

**D1. Chi-squared responder-group fit (code existed, real data now wired in).**
`population_similarity.chi2_fit_to_data()` tests which model's responder-group
composition (the same 7-group taxonomy as section A/B) best matches real
data -- but until today it had only ever been run against a synthetic
placeholder. `cross_model_vs_experiment/extract_experiment_group_counts.py`
now extracts real counts from the experimental repo's tidy responsiveness
tables; `group_counts/{expert,reversal_pre,reversal_post}.json` are generated
and ready. A scipy-free manual preview (see that README) ranks classif_rl
("SSL + RL") as the best shape-match to real expert-phase data, rl_only
second, classif_rl_readout_only worst -- rerun the real script in an env with
scipy for the proper figure + p-values. The reversal-phase comparison
(`reversal_pre.json` / `reversal_post.json`) is extracted but not yet run
against the model side.

**D2. Population-activity overlay (not started).** Plot model
`population_activity_per_stim` directly against the experimental
`draw_expert_mean`/`draw_reversal_mean` trace on shared axes, using the ported
`style.py` stimulus colors and pre/post convention -- a qualitative
"does the model reproduce the trial-averaged activity profile" figure. Doesn't
exist in either repo.

**D3. Full experimental pipeline on model activations (not started, biggest
scope item).** Beyond responder-proportions and population activity, the
experimental analysis includes decoding (stimulus/value/context, cross-phase),
dimensionality (PCA, participation ratio, eigenspectrum), and TDR. Running
these on model activations needs a decision: reshape model outputs into the
same tidy format the experimental `extract_*.py` scripts expect and reuse them
unmodified, or reimplement against `cxval.analysis`'s existing equivalents
(`compute_unit_tuning`, decoding functions, `geometry.py`'s PCA/participation-ratio
code, which already covers some of this). Needs scoping before starting.

## Style unification

`unified_figures/style/` ports the experimental repo's `transfer.mplstyle` +
stimulus color palette + pre/post `condition_style()` convention verbatim (see
that folder's docstrings for exact source lines). It is not yet wired into
`figures.py` / `reversal_analysis.py` in sections A-C -- those keep using
their own `code/figure_style.mplstyle` (larger fonts, PDF+PNG, no tag stamp)
until you decide whether to restyle already-reviewed figures now or later.
New figures (D2 especially) should use the ported style from the start.

## Known issues / decisions needed

- `results/transfer_final/` (inside `context-value-RNNs/results/`) is a
  stale, less-complete duplicate of the repo-root `transfer_final/` (missing
  `readout_only` variants and all reversal figures) -- per your call, treating
  the repo-root copy as canonical and leaving this one untouched.
- `reversal_study/reproducibility/training/` is empty, despite its own README
  describing `train_model.py`/`train_reversal.py`/`run_all.sh` living there.
  The real, actively-used copies live at repo-root `scripts/16_06_26_*.py` (the
  `run_*_full.sh` scripts all call those paths directly). `unified_figures/reversal/jobs/`
  symlinks to the real location so this folder works today, but the
  reproducibility bundle itself should probably be fixed before any future
  move to `~/Documents/transfer`.
- `figures.py`/`reversal_analysis.py`/etc. exist as identical duplicates in
  both `reversal_study/code/` and repo-root `scripts/16_06_26_*.py` (confirmed
  byte-identical for the core figure scripts) -- harmless today but two
  copies to keep in sync by hand; `recovery_time.py`, `terminal_rpe.py`,
  `reversal_onset_probe.py`, `figure_config.py` only exist in
  `reversal_study/code/`, not mirrored to `scripts/`.
- `chi2_fit_to_data` has never been run against real data before now (D1).
- `squash_1p3_full` and `reward_scale_dose_response` are incomplete/unrun (C).
