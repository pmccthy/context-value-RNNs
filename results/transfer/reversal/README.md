# Reversal study

A follow-up experiment on the three final 3s1c models (see the main `transfer_final`
bundle). Each trained model is continue-trained on a **contingency reversal** — the 0%
and 100% reward mappings are swapped (`value_matrix [0,.5,1] → [1,.5,0]`; the 50%
stimulus and all stimulus *inputs* are unchanged) — warm-started from its trained
weights. We ask (a) does each model class recover pre-reversal performance, and (b) how
does its representation reorganise.

Self-contained: it carries its own copy of the trained (pre-reversal) models, so nothing
here depends on `transfer_final`. Two reversal budgets are included: **2,500** and
**5,000** continue-training trials.

## Layout

```
reversal_study/
  README.md
  code/                          reads only figure data / histories
    reversal_analysis.py         recovery + representation-remap figures
    population_similarity.py     per-neuron pre/post change + group chi-squared fit to data
    rsa.py                       representational similarity analysis (RDM heatmaps)
    seed_groups.py               seed-by-seed recovery + recovered-vs-failed comparisons
    figures.py, figure_config.py, figure_style.mplstyle   (shared figure library)
  reproducibility/               regenerate the data (needs PyTorch + the cxval snapshot)
    cxval/                       source snapshot
    training/  train_reversal.py, run_reversal_all.sh
    inference/ run_inference.py
    requirements.txt
  results/
    model_runs/                  pre-reversal models (model.pt + config.json + history.json)
    figure_data/                 pre-reversal figure data (pkl/csv/npz)
    reversal_2500/               2,500-trial reversal
      model_runs_reversal/       reversed models + history.json (recovery curve, probes)
      figure_data_reversal/      post-reversal figure data
      figures/                   all reversal figures (all seeds)
      figures_recovered/         same, restricted to seeds that recovered (>=80%)
      figures_seed_groups/       seed-by-seed + recovered-vs-failed comparisons (see below)
    reversal_5000/               5,000-trial reversal (same sub-layout)
    population_similarity/       cross-model / vs-data comparisons (see below)
      reversal_2500/, reversal_5000/   per-neuron pre/post plots for that budget
      group_fit/                       responder-group chi-squared fit to data
    rsa/                          representational similarity analysis (see below)
```

## Key findings (30 seeds)

- **Recovery is bimodal per seed** — a seed either fully recovers (~100% of pre-reversal
  reward) or fails (~20%). Among *recovered* seeds all three classes reach ~100%.
- **The class difference is the fraction of seeds that recover**, and longer training
  rescues more of the classification models (readout-only barely benefits):

  | class | recovered ≥80% @2,500 | @5,000 |
  |---|---|---|
  | RL only | 28/30 | 30/30 |
  | SSL + RL | 16/30 | 23/30 |
  | SSL + RL (readout-only) | 12/30 | 11/30 |

- **Representation remap** (`reversal_pref_transition`): SSL+RL and readout-only stay
  **diagonal** (units keep stimulus-*identity* tuning; value is re-read in the readout —
  readout-only is nearly frozen), while RL-only goes **anti-diagonal** at the value
  extremes (the code itself re-sorts to the swapped values).

## Reproduce

From the bundle root. Set `B` to the budget you want.

```bash
pip install -r reproducibility/requirements.txt
B=results/reversal_2500        # or results/reversal_5000 (pass N_TRIALS=5000 below)

# 1. reverse-train every model (warm-started from results/model_runs/)
RUNS=results/model_runs OUT=$B/model_runs_reversal N_TRIALS=2500 \
     bash reproducibility/training/run_reversal_all.sh
# 2. inference on the reversed models
python reproducibility/inference/run_inference.py --runs $B/model_runs_reversal --out $B/figure_data_reversal
# 3. standard per-model figures on the reversed models
python code/figures.py --data $B/figure_data_reversal --out $B/figures
# 4. reversal-specific figures (recovery, remap, timelines, pre/post bars & heatmaps)
python code/reversal_analysis.py --pre results/figure_data --post $B/figure_data_reversal \
       --reversal-runs $B/model_runs_reversal --pre-runs results/model_runs \
       --post-runs $B/model_runs_reversal --out $B/figures
# 5. recovered-seeds-only version (+ reversal_recovered_proportion), restrict to >=80% recovery
python code/reversal_analysis.py --pre results/figure_data --post $B/figure_data_reversal \
       --reversal-runs $B/model_runs_reversal --pre-runs results/model_runs \
       --post-runs $B/model_runs_reversal --min-recovery 0.8 --out $B/figures_recovered
```

The combined original→reversal timelines need the pre-reversal models to carry
training-time probes (they do here — `results/model_runs/*/history.json` with `probe_*`).

## Figures

In each `results/reversal_<budget>/figures/`: `reversal_recovery`,
`reversal_recovery_fraction` (% of pre-reversal reward), `reversal_pref_transition`,
`reversal_tuning_correlation`, `reversal_bars_*` (pre vs post), `reversal_timeline_*`
(metric over original→reversal on a trials axis, reversal marked), `reversal_heatmaps_prepost`,
plus the standard per-model figures. `figures_recovered/` is the same set restricted to
recovered seeds, plus `reversal_recovered_proportion` (per-class count).

## Population-level comparisons (`population_similarity.py`)

Two analyses aimed at quantifying representations for cross-model (and eventually
model-vs-real-data) comparison, at the level of individual units rather than
model-wide averages:

**1. Per-neuron pre/post reversal change.** For every hidden unit: the Pearson
correlation between its pre- and post-reversal tuning vector (response to the 3 stimuli)
and its mean signed activity difference (post − pre). High correlation + small
difference = tuning shape and level both preserved; low/negative correlation = the
unit's tuning reorganised. `reversal_tuning_correlation` in `reversal_analysis.py`
already reports the per-model *mean* of this; here it's broken out per unit and plotted
(`population_neuron_scatter`, `population_neuron_distributions`, `population_diff_summary`).
Confirms the earlier finding at the single-unit level: RL-only units have a
correlation distribution centred near/below 0 (tuning reorganises), while SSL+RL and
readout-only sit near +1 (tuning preserved, readout-only almost perfectly).

**2. Responder-group chi-squared fit to data.** Each model's units are already
classified into the 7 responder groups (`figures.py`'s `GROUPS`: the 3 exclusive types,
3 pairwise, all-three). Pooling counts over seeds, a chi-squared goodness-of-fit test
(data proportions rescaled to each model's N) scores how close each model's group
*composition* is to a real dataset's. The model with the lowest χ² (dof is identical
across models, so this also means highest p) is the representation least
distinguishable from the data on this criterion — `population_group_chi2` ranks the
three models and reports p-values; `population_group_proportions_vs_data` shows the
composition bars side by side.

**No real neural data yet** — `group_fit/` currently compares against a clearly-labelled
FAKE placeholder dataset (a multinomial draw, seeded, printed to the console each run).
Swap in real counts with `--data-counts group_counts.json`:
```json
{"0%-only": 41, "50%-only": 33, "100%-only": 52, "0% & 50%": 12,
 "0% & 100%": 9, "50% & 100%": 14, "all three": 22}
```

Regenerate:
```bash
python code/population_similarity.py --pre results/figure_data \
       --post results/reversal_5000/figure_data_reversal \
       --group-data results/figure_data \
       [--data-counts group_counts.json] \
       --out results/population_similarity/reversal_5000
```

## Representational similarity analysis (`rsa.py`)

Condition x condition representational dissimilarity matrices (RDMs), the standard RSA
tool for comparing representational *geometry* rather than raw activity levels — the
natural next step for eventually scoring model-vs-real-data fit.

Conditions are stimulus (3) x trial-time bin (`n_align` = ITI + stim + outcome, stim-major
block order), so each condition is "this stimulus, at this point in the trial". For each
model, its population-activity vector per condition is built by pooling units over all its
seeds; `RDM[i, j] = 1 - Pearson_r` between conditions' population vectors, correlated
across the unit axis (not neuron-by-neuron, not over trials — see the conceptual
discussion earlier in this project for the crossnobis/trial-based upgrade).

  - `rsa_rdm_heatmaps` — TIME-RESOLVED (conditions = stimulus × trial-time, 33×33): one
    RDM per model, shared colour scale, thick lines mark stimulus blocks and dotted lines
    the ITI|stim|outcome boundary within each block. The bright off-diagonal blocks show
    each model represents the 3 stimuli as fairly distinct population patterns; the
    within-block structure (bright square vs. smooth gradient) shows how sharply the
    ITI/stim/outcome periods are distinguished.
  - `rsa_rdm_heatmaps_stim` — STIMULUS-ONLY (conditions = the 3 stimuli, 3×3, using each
    unit's average stim-window response — the same "first-order RSA" idea collapsed over
    trial-time, and the more standard/simple RSA form). Values annotated. This view makes
    a real qualitative difference between classes visible: RL-only's 50%/100% dissimilarity
    is tiny (0.11 — those two values drive near-identical population patterns) while 0% is
    far from both; SSL+RL is intermediate; readout-only separates all three sharply
    (~1.2, near-orthogonal codes for each stimulus identity).
  - `rsa_second_order` / `rsa_second_order_stim` — model x model heatmap of the Spearman
    correlation between each pair of models' RDMs (upper triangles), for each RDM type.
    Time-resolved: SSL+RL and RL-only share the most representational geometry (0.79),
    readout-only diverges most from RL-only (0.53) — consistent with readout-only's RL
    gradients never reaching the backbone. Stimulus-only has just 3 upper-triangle values
    per RDM, so its second-order correlations are necessarily coarse (only take values
    like 0.5/1.0) — treat that one as a low-powered sanity check, not a precise score.

Regenerate (defaults to the pre-reversal / final models; pass `--data` at any other
figure_data, e.g. a reversal budget's `figure_data_reversal`, to build RDMs for that
representation instead):
```bash
python code/rsa.py --data results/figure_data --out results/rsa
```

Once you have real neural-data condition responses, build its RDM the same way (same
condition order) and add it as a `data` column to `rsa_second_order` — the model with the
highest correlation to it is the best second-order-RSA fit, complementing the
`population_similarity.py` chi-squared score.

## Seed-by-seed & recovered-vs-failed comparisons (`seed_groups.py`)

Recap of the recovery metric: `recovered_fraction` = a seed's post-reversal mean reward
divided by that SAME seed's pre-reversal mean reward (both from `history.json`).
**Recovered** = `recovered_fraction >= 0.8` (the threshold used throughout), else
**failed**. This is the split behind `figures_recovered/` and the recovery-proportion
table above.

  - `seed_recovery_strip` — every seed's raw `recovered_fraction` as a point (filled =
    recovered, open = failed), per class, with the 80% threshold marked. Shows the
    bimodality directly (clustered near 100% or near ~20%, essentially nothing in
    between) rather than only the summary count.
  - `seed_group_tuning_correlation` / `seed_group_activity_diff` — re-runs
    `population_similarity.py`'s per-neuron pre/post metrics SEPARATELY on each class's
    recovered seeds and its failed seeds (via `reversal_analysis.filter_fd`), plotted as
    violins with a Mann-Whitney U p-value per class. At 2,500 trials: RL-only's failed
    seeds have a distinctly negative mean tuning correlation (~-0.4) vs its recovered
    seeds (~0, p=0.01) — its rare failures look like partial anti-diagonal remapping
    that didn't fully complete. Readout-only shows no difference between groups
    (p=0.84) — tuning is preserved either way, so recovery there is decided elsewhere
    (in the readout weights, which this doesn't probe). All three classes show a real
    but small activity-level-shift difference between groups (p ranging 1e-7 to 1e-29).
  - `seed_group_responder_groups` — post-reversal responder-group composition, recovered
    vs failed, solid vs hatched bars per class.
  - `seed_group_rdm_stim` — 2×3 grid (recovered / failed rows × class columns) of the
    post-reversal stimulus-only RDM (see `rsa.py`), so the representational geometry of
    the two outcome groups can be compared directly.
  - `seed_group_heatmaps` — the same 2×3 grid but as raw per-unit tuning heatmaps
    (activation, not RDM), one representative seed per cell.
  - `seed_group_heatmap_montage_<recovered|failed>_<model_type>` — six montages, each
    tiling EVERY seed's tuning heatmap within that (outcome group, class), reusing
    `figures.py`'s `heatmap_montage`, so you can see the full within-group seed-to-seed
    spread rather than one example. Fixed a real bug surfaced by this: `heatmap_montage`
    sized its figure only by seed count, so small groups (e.g. RL-only's 2 failed seeds)
    produced a too-short figure where the title collided with the panels — now a fixed
    margin is reserved regardless of group size. Also, `heatmap_montage`'s default title
    only named the model, not which seed subset it was filtered to — it now takes a
    `title=` override, and these montages say **RECOVERED SEEDS** / **FAILED SEEDS**
    explicitly (previously that was only encoded in the filename).
  - `seed_vigour_curves` — EVERY seed's mean-vigour trajectory, overlaid on the same
    axes. Matches the standard convention used everywhere else in this bundle
    (`figures/reversal_timeline_vigour`, `reversal_analysis.py`'s `fig_combined_timeline`)
    as closely as possible: each stimulus keeps its usual `STIM_COLORS` colour, legend
    "stimulus (pre → post)" via `REV_LABELS`, and ORIGINAL TRAINING + REVERSAL are drawn
    on one continuous trials axis with the reversal onset marked by a dashed line. The
    only real deviations, both necessary: each seed is its own line rather than a mean ±
    SEM band (the point is to see individual seeds), and the outcome-group split
    (recovered vs failed) is by ROW rather than colour, since colour was already spoken
    for by stimulus — columns are still model class, as everywhere else. This is the
    clearest single view of what "recovery" looks like as a process: for the 0%→100% and
    100%→0% stimuli, recovered seeds swing cleanly to the new contingency after the
    dashed line while failed seeds visibly plateau partway or drift back; the 50%→50%
    (unchanged) stimulus stays noisy around the same level throughout, before and after,
    for both outcomes, as expected since nothing changed for it.

**Style-sheet bug, now fixed:** `figures.py`'s `DEFAULT_STYLE` constant pointed at
`16_06_26_figure_style.mplstyle` (the filename in `scripts/`), but this bundle's copy is
named `figure_style.mplstyle` (no prefix) — a straight `cp` of the script into `code/`
silently broke it, and `main()`'s `if style_file.exists(): plt.style.use(...)` had no
`else`, so every `seed_group_*` figure regenerated in between quietly fell back to plain
matplotlib defaults (small fonts, visible top/right spines) with no warning. Fixed by
making `DEFAULT_STYLE` check both filenames and by making the fallback print a loud
warning instead of silently doing nothing. All `seed_group_*` figures were regenerated
after the fix; `population_similarity.py`/`rsa.py` outputs predate the breakage and were
already correct.

Small-sample caveat: group sizes are very uneven for some classes (e.g. RL-only has only
2 failed seeds at 2,500 trials and 0 at 5,000 — those panels/tests are skipped or
low-powered automatically). **The 5,000-trial budget is the default** (RL-only having
zero failures there is itself part of the finding — with enough continue-training its
rare 2,500-trial failures resolve); the 2,500-trial budget gives better-balanced groups
if you want that comparison instead, and both are pre-generated under
`results/reversal_<budget>/figures_seed_groups/`.

Regenerate (defaults to the 5,000-trial budget; pass `--post`/`--reversal-runs`/`--out`
to target 2,500 instead):
```bash
python code/seed_groups.py --pre results/figure_data --pre-runs results/model_runs \
       --post results/reversal_5000/figure_data_reversal \
       --reversal-runs results/reversal_5000/model_runs_reversal \
       --out results/reversal_5000/figures_seed_groups
```

## Pilot: does flooring vigour above 0 fix the stuck-at-zero failures? (`results/vigour_floor_pilot/`)

**Motivation.** Failed seeds' vigour sits at exactly 0.0 for the reversed-to-100% and
reversed-to-0% stimuli, never recovering. The likely mechanism: the score-function
(REINFORCE) policy samples `Normal(mean, action_std=0.05)` then hard-clamps to `[0,1]`. If
`mean` drifts negative enough that ~all samples clip to exactly 0, the executed action
stops varying with `mean` — the score-function gradient signal (`action − mean`)
degenerates, and the policy has no path back out. Two fixes were proposed: widen
`action_std`, or floor the clamp above 0 so the executed action (and its reward/cost
feedback) never fully degenerates.

**What was built.** `cxval/vigour.py`'s `VigourActorCritic` gained a `min_vigour`
attribute (default 0.0 = unchanged behaviour); `squash()` now clamps to
`[min_vigour, 1]` instead of a hard-coded `[0, 1]`. This also fixed a second, more direct
bug found while implementing it: the actual training rollout (`train_vigour`'s inner
loop) sampled the action and clamped it with a **separate hard-coded `.clamp(0.0, 1.0)`
call that bypassed `squash()` entirely** — so the floor would have been a no-op for
training specifically if left unfixed. Exposed as `--min-vigour` on `train_model.py`
(flows through `config.json`'s `"train"` dict, so `train_reversal.py` and
`run_inference.py` inherit it automatically for warm-started/evaluated runs — both also
needed a matching fix, since each reconstructs a fresh `VigourActorCritic` from a bare
state_dict for deterministic eval and would otherwise silently ignore the floor a model
was actually trained with).

**Pilot: 5 seeds x 3 classes, `--min-vigour 0.1`, 2,500-trial original training +
2,500-trial reversal** (seeds 90-94, kept separate from the main study's 42-71 range):

| class | recovered ≥80% (pilot, n=5) | recovered ≥80% (baseline min_vigour=0, n=30, @2,500) |
|---|---|---|
| RL only | 5/5* | 28/30 |
| SSL + RL | 1/5 | 16/30 |
| SSL + RL (readout-only) | 2/5 | 12/30 |

\* one of these five (seed 92) never learned in the first place — pre-reversal reward
was already near-zero (0.007 vs a healthy ~0.05), because it got stuck at the *new*
floor (vigour = 0.10 for all three stimuli) during ORIGINAL training. Its 96% "recovery"
is recovering back to that same degenerate near-zero baseline, not real performance.

**Reading:** the floor does not straightforwardly fix the problem — it relocates the
same clipping trap rather than removing it (samples can just as easily saturate at 0.10
as at 0.0), and it introduced a failure mode that essentially didn't exist in the
baseline study: ORIGINAL (pre-reversal) training getting stuck at the floor, not just
the reversal step. In this small pilot, SSL+RL's post-reversal recovery rate looks
*worse* than baseline (1/5 vs a 53%-equivalent rate), while RL-only and readout-only are
roughly in the same range as baseline, within what 5 seeds can distinguish. `seed_vigour_curves`
(below) shows this directly: several lines sit exactly flat at 0.10 for an entire
stimulus, in both the pre- and post-reversal segments — the same qualitative "stuck"
signature as the original 0.0-floor failures, just shifted up.

**Caveats:** n=5/class is a pilot, not a replacement for the 30-seed study — treat the
proportions as suggestive, not definitive. The `action_std`-widening alternative
(untried) remains a live, likely-more-promising hypothesis: it increases exploration
noise everywhere rather than moving the saturation point, so it doesn't relocate the
trap, it actually reduces how easily the policy saturates against either boundary.

Layout: `results/vigour_floor_pilot/{model_runs, figure_data, model_runs_reversal,
figure_data_reversal, figures}` (`figures/` has `seed_recovery_strip` and
`seed_vigour_curves`, generated with `code/seed_groups.py`'s functions directly since
n=5/class is too small for the chi-squared/Mann-Whitney comparisons built for n=30).

Reproduce:
```bash
for MT in classif_rl rl_only classif_rl_readout_only; do
  for S in 90 91 92 93 94; do
    python reproducibility/training/train_model.py --model-type $MT --seed $S \
        --n-trials 2500 --min-vigour 0.1 --probe-every 20 \
        --out results/vigour_floor_pilot/model_runs/$MT/seed$S
    python reproducibility/training/train_reversal.py \
        --run results/vigour_floor_pilot/model_runs/$MT/seed$S \
        --out results/vigour_floor_pilot/model_runs_reversal/$MT/seed$S \
        --n-trials 2500 --probe-every 20
  done
done
python reproducibility/inference/run_inference.py --runs results/vigour_floor_pilot/model_runs \
       --out results/vigour_floor_pilot/figure_data
python reproducibility/inference/run_inference.py --runs results/vigour_floor_pilot/model_runs_reversal \
       --out results/vigour_floor_pilot/figure_data_reversal
```

**Next step: full 30-seed run.** `code/run_min_vigour_full.sh` runs the same
`min_vigour=0.1` intervention across the full 30 seeds (42-71, same seeds as the
baseline study, for direct seed-for-seed comparability) and regenerates all the usual
bundle plots, including a real (not terminal-model-only) `seed_rpe_curves` training-time
trajectory (see "Reward-prediction error" below). It's a long CPU run (90 training
calls x 2 phases) — meant to be launched in the background:
```bash
cd reversal_study
nohup bash code/run_min_vigour_full.sh > /tmp/min_vigour_full.log 2>&1 &
tail -f /tmp/min_vigour_full.log
```
Output lands in `results/min_vigour_0p1_full/`. The script is safe to re-run — each
step skips seeds/classes it already completed.

## Reward-prediction error (RPE)

`cxval/vigour.py`'s `infer_rpe()` computes the actor-critic's TD error,
`delta_t = r_t + gamma*V(s_t+1) - V(s_t)`, per stimulus, from one deterministic
rollout — the same rollout cost as `infer_vigour()`, no extra model calls. It's now
wired into both training scripts' `make_probe()` (`train_model.py`, `train_reversal.py`),
so any run trained from now on logs `probe_rpe` in `history.json` at the same cadence as
`probe_vigour`, and `seed_groups.py`'s `main()` calls `fig_seed_curves(..., key="rpe",
stem="seed_rpe_curves")` automatically alongside the vigour curves — same rows=outcome
group / columns=model-class / `STIM_COLORS` format, just RPE instead of vigour.

**Existing (baseline, `min_vigour=0`) runs don't have this.** They only saved the FINAL
model per phase, not periodic checkpoints, so there's no way to recover what RPE looked
like *during* the original training after the fact — the value estimate RPE is defined
against needs the model's live weights at each point, and those weights are gone.
`code/terminal_rpe.py` computes the best available substitute without retraining:
repeated deterministic evaluations of the frozen final pre-reversal model (on the
original task) and the frozen final post-reversal model (on the reversed task), so you
get a "terminal RPE" reading rather than a learning curve — point-to-point variation
along its x-axis is pure sampling noise from the stochastic trial draws, not a training
trend (the figure's subtitle says this explicitly). It reuses `seed_groups.py`'s
`fig_seed_curves` unmodified (writing synthetic per-seed `history.json` files so the
existing, already-verified plotting code can be reused as-is) via a new `phase_note`
parameter that overrides the "original training → reversal" wording for this non-training
case.

Output: `results/reversal_5000/figures_seed_groups/terminal_rpe_curves.png`. Regenerate:
```bash
python code/terminal_rpe.py --pre-runs results/model_runs \
       --post-runs results/reversal_5000/model_runs_reversal \
       --out-runs results/terminal_rpe --out results/reversal_5000/figures_seed_groups
```
(`--skip-plot` + `--seed-min`/`--seed-max` let you split the compute across multiple
calls if needed; already-computed seeds are skipped automatically on re-run.)

`seed_groups.py`'s `main()` now also calls `fig_seed_curves_avg()` for both `vigour`
and `rpe` — the group MEAN ± SEM companion to the seed-resolved spaghetti plots (same
rows=outcome-group / columns=model-class / `STIM_COLORS` layout, band = SEM across
seeds in that group instead of one line per seed). Outputs: `seed_vigour_curves_avg`,
`seed_rpe_curves_avg` (the latter only where `probe_rpe` exists, i.e. not for the
baseline study's pre-reversal runs — same limitation as above). These make the
group-level pattern easier to read off than the individual-seed version, at the cost of
hiding seed-to-seed heterogeneity — use both together.

The real (`min_vigour=0.1`) `seed_rpe_curves` / `_avg` figures show a genuine
reversal-learning RPE signature that the terminal-model version categorically couldn't:
a sharp negative RPE dip for the newly-devalued stimulus right at reversal onset,
decaying back to ~0 as the agent relearns — and, for the RECOVERED group, a mirrored
positive RPE bump for the newly-valued stimulus. For the FAILED group the same initial
dip appears but decays to ~0 for a different reason (the value function collapses to
predicting ~nothing, so there's nothing left to be wrong about) — worth keeping in mind
when reading "RPE near zero" as "prediction is good."

## Time to recovery (`code/recovery_time.py`)

`recovered_fraction` (used throughout above) is a single pre/post endpoint ratio — it
says WHETHER a seed recovered, not HOW FAST. `recovery_time.py` adds a time-resolved
version: the first point (in trials) a SMOOTHED curve crosses `--thr` (default 0.8) of
the seed's own pre-reversal reference level, in two independent flavours:

- **reward-based** — uses the fine-grained PER-UPDATE `mean_reward` history (every
  optimizer update, not just every probe) against `pre_reversal_reward`, smoothed with
  a rolling mean over `--reward-smooth` updates (default 51).
- **vigour-based** — since physical stimulus identity (0/1/2) is unchanged by the
  reversal, only which one is high/low value, define the "reversed-value span"
  `S_t = vigour[stim 0] - vigour[stim 2]` at each post-reversal probe (positive once
  correctly relearned) as a fraction of the seed's own converged PRE-reversal span
  `vigour_pre[stim 2] - vigour_pre[stim 0]`, smoothed over `--vigour-smooth` PROBES
  (default 3, since probes are already coarser-grained than updates).

These need not agree — reward and vigour don't have to recover in lockstep — which is
exactly what comparing them is for. Output: `recovery_time.png` (two panels, reward- vs
vigour-based, jittered strip + median tick per model class, `n reached / n total`
annotated on the x-tick labels).

On the `min_vigour=0.1` full run, reward-based time-to-recovery shows a cluster of
near-instant (~0 trial) "recoveries" for SSL+RL and the readout-only variant that
doesn't appear in the baseline study — these are the same degenerate seeds flagged in
the pilot section above (pre-reversal reward already collapsed near zero because the
seed got stuck at the floor during ORIGINAL training), trivially clearing the 80%
threshold because the threshold itself is near zero. Not a bug in the metric — a
real artefact of the floor, worth excluding by eye when comparing medians.

Regenerate for either study:
```bash
python code/recovery_time.py --pre-runs results/model_runs \
       --post-runs results/reversal_5000/model_runs_reversal \
       --out results/reversal_5000/figures_seed_groups
python code/recovery_time.py --pre-runs results/min_vigour_0p1_full/model_runs \
       --post-runs results/min_vigour_0p1_full/model_runs_reversal \
       --out results/min_vigour_0p1_full/figures_seed_groups
```

## Critic value estimate + the "critic collapse" hypothesis

Why some seeds stay stuck even with `min_vigour=0.1`: the floor fixes the *action*-side
degeneracy (executed vigour can no longer literally saturate at exactly 0), but the
*critic* can independently converge to an accurate, self-consistent LOW value
prediction for a stimulus the policy has stopped exploring -- at which point the TD
error (RPE) collapses to ~0 not because the value is right in an absolute sense, but
because it correctly predicts what the (stuck) policy is actually earning. With near-
zero RPE, the policy gradient (scaled by RPE/advantage) has nothing left pushing it
toward the better, unexplored policy.

`cxval/vigour.py`'s `infer_value()` (mirrors `infer_rpe`, same rollout cost) exposes the
critic's raw V(s) prediction per stimulus, now wired into both training scripts'
`make_probe()` alongside vigour/RPE, so any run from now on logs `probe_value` and
`seed_groups.py`'s `main()` produces `seed_value_curves` / `_avg` automatically.

For the ALREADY-COMPLETED `min_vigour=0.1` job, `terminal_rpe.py` was extended to also
compute a terminal (frozen-final-model) value estimate the same way it does terminal
RPE -- `terminal_value_curves.png` in `results/min_vigour_0p1_full/figures_seed_groups/`.
It directly confirms the hypothesis: for FAILED seeds, the critic's V(s) for the
newly-valuable stimulus stays low (~0.2-0.3) post-reversal, essentially unchanged from
its pre-reversal (correctly low) level -- the critic never came to expect anything
better there. For RECOVERED seeds, V(s) for that same stimulus correctly rises to
~1.1-1.4 post-reversal, mirroring what the other stimulus's value used to be
pre-reversal. Regenerate:
```bash
python code/terminal_rpe.py \
       --pre-runs results/min_vigour_0p1_full/model_runs \
       --post-runs results/min_vigour_0p1_full/model_runs_reversal \
       --out-runs results/terminal_value_minvig \
       --out results/min_vigour_0p1_full/figures_seed_groups
```

## Pilot: does widening action_std (instead of/alongside the floor) fix it?

Since the critic-collapse mechanism is fundamentally an exploration problem (the critic
never observes a better outcome because the policy stopped generating it), widening the
Gaussian policy's `action_std` is a more direct fix than flooring the clamp boundary --
it keeps higher-vigour actions getting sampled by chance regardless of where the
readout mean has drifted, rather than just moving where the clamp sits.

**5 seeds x 3 classes, `--action-std 0.15` (3x baseline's 0.05), `min_vigour=0` (no
floor, to isolate this from the earlier intervention)**, seeds 120-124, `results/action_std_pilot/`:

| class | recovered ≥80% (action_std=0.15, n=5) | baseline (action_std=0.05, n=30) |
|---|---|---|
| SSL + RL | **5/5 (100%)** | 22-23/30 (~73-77%) |
| RL only | 5/5 (100%) | 28-30/30 (already ~ceiling) |
| SSL + RL (readout-only) | 2/5 (40%) | 11/30 (~37%) |

SSL+RL goes from its baseline ~75% recovery rate to a clean 5/5 in this small pilot --
consistent with the exploration-based mechanism directly targeting what was actually
broken for that class. RL-only was already near-ceiling at baseline so there's no room
to see an effect. Readout-only is unmoved (still ~40%, matching baseline), which fits a
DIFFERENT, non-exploration explanation for that class: since RL gradients never reach
the recurrent units there (`detach_readout=True`), only the linear readout can adapt
post-reversal, so failures there are more likely a representation-geometry problem
(whether the frozen hidden code is even linearly re-mappable) than an exploration one
-- more `action_std` can't fix a representation the readout has no way to bend.

**Caveats:** n=5/class, same small-sample caution as the earlier pilot. `action_std`
wasn't swept -- 0.15 was one reasonable choice, not tuned.

Output: `results/action_std_pilot/figures/` (`seed_recovery_strip`, `seed_vigour_curves`,
`seed_value_curves`).

## How "recovered" / "pass" / "fail" is defined (recap)

`recovered_fraction` = post-reversal mean reward / that SAME seed's own pre-reversal
mean reward. Both come from `train_reversal.py`'s `eval_reward()` -- a deterministic
evaluation (`model.squash(mean)`, not sampled) of 8 episodes x 200 trials, reward
averaged per active timestep: pre-reward is the trained model on the ORIGINAL task,
post-reward is the reversal-trained model on the REVERSED task. A seed is **recovered**
("pass") if `recovered_fraction >= 0.8` (the `--thr` default everywhere), else
**failed**. It's normalized per-seed specifically so classes with different absolute
reward levels are still comparable on the same 0-100% scale. This is reward-based, not
vigour-based -- `recovery_time.py`'s vigour-based variant (see above) is a different,
independently-computed criterion using the reversed-value vigour span instead.

## Seed-resolved selectivity/activity summaries + selectivity-remapping alluvial diagrams

Two more per-seed summary plots (companions to `seed_group_heatmaps`, which only shows
one representative seed's heatmap per cell): `seed_responder_proportions` (% of
responsive units in each of the 7 responder groups -- 0%-only, 50%-only, ..., all
three -- one point per seed) and `seed_population_activity` (population activity per
stimulus, one point per seed), both filled=recovered/open=failed, one panel per model
class. Unlike the existing pooled bar versions (`seed_group_responder_groups`), these
show the actual seed-to-seed spread underneath the summary, and let you check by eye
whether pass/fail seeds separate along either metric.

`seed_group_alluvial` / `_pooled` / `_montage_<group>_<class>`: a Sankey-style flow
diagram of how units' selectivity CATEGORY changes across the reversal -- silent (not
significantly responsive to any stimulus, t-test) vs. preferred-stimulus (argmax
tuning) among units that ARE responsive. Left column = pre-reversal category
composition, right = post, ribbons coloured by the PRE-reversal category (so you can
trace e.g. how much of "pref 100%" flowed into "pref 0%" -- the anti-diagonal remap
`reversal_pref_transition` already summarises numerically, just visualised as flow
rather than a matrix, and now WITH a silent category instead of excluding non-responders).
Three versions: `_alluvial` (one representative seed per outcome-group x class cell, 2x3
grid), `_alluvial_pooled` (same grid, but pooling every unit from every seed in that
cell -- combining neurons across runs), and `_alluvial_montage_<group>_<class>` (every
individual seed's own diagram, tiled, six files total). On the `action_std_0p15_full`
data this makes the readout-only class's structural bottleneck visually obvious: its
ribbons are almost entirely diagonal (near-zero remapping) in BOTH outcome groups,
consistent with RL gradients never reaching the recurrent units there, vs. SSL+RL and
RL-only showing a clear anti-diagonal flow (0%<->100% swapping) in the recovered group.

All five are now wired into `seed_groups.py`'s `main()`, so they get generated
automatically for any study run through the normal pipeline.

**Full 30-seed run — done.** `code/run_action_std_full.sh` scaled this to the full 30
seeds (42-71, same as baseline); output in `results/action_std_0p15_full/`. Since
`infer_value`/`infer_rpe` are wired into `make_probe`, this run logged GENUINE
training-time `probe_vigour`, `probe_rpe`, AND `probe_value` (no terminal-model
workaround needed). Recovery rates (reward-based, `recovered_fraction >= 0.8`):
SSL+RL 27/30 (90%), RL-only 30/30 (100%), SSL+RL readout-only 6/30 (20%). SSL+RL and
RL-only both clearly improve over baseline (~73%/~100%) as the 5-seed pilot predicted.
Readout-only, however, comes out WORSE here (20%) than both its own 5-seed pilot (2/5 =
40%) and the baseline 30-seed rate (~37%) -- i.e. widening the policy noise did not help
(and if anything hurt) the structurally-bottlenecked class, consistent with the
mechanistic story (RL gradients never reach the recurrent units when
`detach_readout=True`, so exploration noise on the readout alone has nothing to leverage
upstream) but a reminder that the 5-seed pilot's 40% was noisy/not fully representative.

Regenerate:
```bash
cd reversal_study
nohup bash code/run_action_std_full.sh > /tmp/action_std_full.log 2>&1 &
tail -f /tmp/action_std_full.log
```

Reproduce the pilot:
```bash
for MT in classif_rl rl_only classif_rl_readout_only; do
  for S in 120 121 122 123 124; do
    python scripts/16_06_26_train_model.py --model-type $MT --seed $S \
        --n-trials 2500 --action-std 0.15 --probe-every 20 \
        --out results/action_std_pilot/model_runs/$MT/seed$S
    python scripts/16_06_26_train_reversal.py \
        --run results/action_std_pilot/model_runs/$MT/seed$S \
        --out results/action_std_pilot/model_runs_reversal/$MT/seed$S \
        --n-trials 5000 --probe-every 20
  done
done
python scripts/16_06_26_run_inference.py --runs results/action_std_pilot/model_runs \
       --out results/action_std_pilot/figure_data
python scripts/16_06_26_run_inference.py --runs results/action_std_pilot/model_runs_reversal \
       --out results/action_std_pilot/figure_data_reversal
```

## Mixed-selectivity alluvial (8-category Sankey)

The plain `seed_group_alluvial*` figures (above) collapse each responsive unit to its
single argmax-preferred stimulus, discarding mixed-selectivity units (responsive to 2 or
all 3 stimuli) into whichever they happen to prefer most. `fig_group_alluvial_mixed` /
`_mixed_pooled` are the same Sankey machinery run over all 8 categories instead: silent +
the 7 responder groups already used elsewhere (`0%-only`, `50%-only`, `100%-only`, `0% &
50%`, `0% & 100%`, `50% & 100%`, `all three`), so a unit that's responsive to two stimuli
gets its own distinct pre/post category rather than being folded into one. Same 2x3 grid
(rows = recovered/failed, cols = model class), ribbons coloured by PRE-reversal category.
`_mixed` uses one representative seed per cell; `_mixed_pooled` pools every unit from
every seed in that cell. (No montage version for the 8-category variant -- the plain
4-category `_alluvial_montage_*` already covers the per-seed view; add one if useful.)

On `action_std_0p15_full`, the pooled mixed view shows the same qualitative pattern as
the plain alluvial -- readout-only's ribbons stay close to the diagonal (little
remapping) in both outcome groups, while SSL+RL and RL-only show clear off-diagonal flow
in the recovered group -- but now resolved into which SPECIFIC mixed-selectivity
categories are involved (e.g. how much of `50% & 100%` units become `0% & 50%` after the
0%<->100% swap, vs. collapsing that into whichever single stimulus dominated).

Output: `seed_group_alluvial_mixed.png`, `seed_group_alluvial_mixed_pooled.png`, wired
into `seed_groups.py`'s `main()` alongside the plain versions.

## Vigour target-matching recovery definition (`code/recovery_time.py`)

A third, independent definition of "recovered," alongside the canonical reward-based
`recovered_fraction` and the vigour reversed-value-SPAN definition (both above). The span
definition only checks that stim-0-vs-stim-2 vigour has flipped sign/magnitude
correctly; it doesn't check that stim 1 (whose value is untouched by the reversal) stays
put, or that the absolute post-reversal levels actually match what pre-reversal vigour
looked like for the stimulus that's now equivalent. Target-matching checks all three
stimuli against an explicit target:

```
target = [vigour_pre[stim 2], vigour_pre[stim 1], vigour_pre[stim 0]]   # mirror swap: 0<->2, 1 fixed
err_t  = mean(|vigour_post[t] - target|)                                # mean abs error, 3 stimuli
```

`err_t` is normalized by the seed's own converged PRE-reversal span `S_pre = vigour_pre[stim
2] - vigour_pre[stim 0]`, NOT by the raw per-stimulus target values. Dividing by the raw
target would blow up: the target for the newly-devalued stimulus (originally 100%, now
0%) is near zero post-reversal, so tiny absolute noise there would register as huge
relative error exactly where the metric most needs to be well-behaved. Normalizing by
the fixed, well-conditioned pre-reversal span sidesteps that. Recovered once the smoothed
`err_t / |S_pre|` first drops to `<= --target-thr` (default 0.2, i.e. within 20% of the
seed's own pre-reversal dynamic range, averaged across all 3 stimuli).

Checked against all three studies (`reversal_5000`, `min_vigour_0p1_full`,
`action_std_0p15_full`): the target-matching definition agrees with the canonical
reward-based pass/fail on all but 1-2 seeds out of ~90 per study (both directions, no
systematic bias) -- reassuring, since it means the simpler reward-based definition used
throughout this README isn't hiding a materially different picture, and the numerical
instability initially flagged as a risk (dividing by a near-zero target) doesn't actually
bite once normalized this way.

Output: `recovery_time.png` gains a third panel (target-match TTR, alongside reward- and
span-based), and a new figure `recovery_definition_comparison.png` -- grouped bars, %
recovered per model class under all three definitions side by side, plus a printed
per-class reclassification count (agree-recovered / agree-failed / reward-only-pass /
alt-only-pass) against the canonical reward-based definition.

Regenerate for any study:
```bash
python code/recovery_time.py --pre-runs results/action_std_0p15_full/model_runs \
       --post-runs results/action_std_0p15_full/model_runs_reversal \
       --out results/action_std_0p15_full/figures_seed_groups --target-thr 0.2
```
