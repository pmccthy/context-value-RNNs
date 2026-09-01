# Reproducibility

Everything needed to regenerate the data in this bundle from scratch. Run the commands
from the **bundle root** (`transfer_final/`).

```bash
pip install -r reproducibility/requirements.txt   # numpy, pandas, matplotlib, torch, scipy, scikit-learn
```

## `cxval/` — source snapshot

The importable package as it was when these results were produced (model, task/env,
training, analysis). The scripts below add this folder to `sys.path` automatically, so
`import cxval...` resolves without installation.

## `training/` — make the models

`train_model.py` trains one model and writes, into `<out>/`:
`model.pt`, `model_init.pt`, **`config.json`** (every effective parameter — training,
task timing, model type), and `meta.txt` (one-line summary).

```bash
python reproducibility/training/train_model.py \
       --model-type classif_rl --seed 42 --out model_runs/classif_rl/seed42
```

`--model-type` is one of `classif_rl`, `rl_only`, `classif_rl_readout_only`; it sets the
two switches that distinguish the arms (`aux_coef` and `detach_readout`). Everything
else defaults to the final config (cost 0.8, init 0.02, readout_fraction 0.5,
action_std 0.05, activity_coef 2.0 at ITI, n_trials 2500, lr 5e-4, gamma 0.9, bptt 40,
hidden 128, batch 32) and is overridable.

`run_all.sh` loops all three model types over a seed range:

```bash
# defaults: seeds 42–71, OUT=../../model_runs
SEEDS="42 43 44" bash reproducibility/training/run_all.sh
```

## `inference/` — make the figure data

`run_inference.py` runs ONE deterministic inference pass per model (default
`8 × 200 = 1600` interleaved trials, ≈530/stimulus; the count is the constant block at
the top of the file) and writes, under `--out`:

- `figure_data.pkl` — self-documenting dict: scalars, per-unit tuning, per-unit
  trial-aligned means, responsiveness (all with `dims` + `coords`).
- `metrics.csv` — the scalar metrics in tidy long form.
- `activations.npz` — per-unit stim tuning as `(3, H)` arrays keyed `"{type}_{seed}"`.
- `time_resolved/<type>_seed<NN>.npz` — the raw per-trial, time-resolved aligned hidden
  activity (the PSTH/grid source).

```bash
python reproducibility/inference/run_inference.py --runs model_runs --out figure_data
```

The pass is resumable (per-model cache under `figure_data/_cache/`); pass
`--budget SECONDS` to compute in chunks and re-run to continue.

## Then the figures

```bash
python code/figures.py --data figure_data --out figures
```
