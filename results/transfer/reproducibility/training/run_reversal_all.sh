#!/usr/bin/env bash
# Continue-train every trained model on the reversal (swap 0% <-> 100%), warm-started.
# Reads trained runs from RUNS (default $HERE/../../model_runs), writes reversed runs
# to OUT (default $HERE/../../model_runs_reversal), mirroring <type>/seed<NN>/.
#
# Usage:  [SEEDS="42 43"] [RUNS=/path] [OUT=/path] [N_TRIALS=5000] [PY=python] bash run_reversal_all.sh
#
# Unlike the previous version, RUNS/OUT default to paths anchored on this
# script's own directory (like run_all.sh does) instead of the caller's CWD,
# and the script refuses to exit "done" if it silently found nothing to do.
set -euo pipefail
PY="${PY:-python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS="${RUNS:-$HERE/../../model_runs}"
OUT="${OUT:-$HERE/../../model_runs_reversal}"
SEEDS="${SEEDS:-$(seq 42 71)}"
N_TRIALS="${N_TRIALS:-2500}"          # reversal continue-training length (trials/episode)
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-0}"   # 0 = off (default); >0 saves a full model
                                             # checkpoint every N updates under
                                             # <out>/checkpoints/ (plus always update 1)
CHECKPOINT_FINE_UNTIL="${CHECKPOINT_FINE_UNTIL:-$CHECKPOINT_EVERY}"  # fine-resolution window
                                             # right after the reversal starts (default:
                                             # fills the first coarse interval)
CHECKPOINT_FINE_EVERY="${CHECKPOINT_FINE_EVERY:-1}"   # fine-window interval (default: every update)

if [ ! -d "$RUNS" ]; then
  echo "ERROR: RUNS directory does not exist: $RUNS" >&2
  echo "       (did you mean to set RUNS=/path/to/model_runs ? no defaults are CWD-relative anymore," >&2
  echo "        so this can no longer fail silently -- but double check the path.)" >&2
  exit 1
fi

n_found=0
n_missing=0
n_done=0
missing_list=()

for MT in classif_rl rl_only classif_rl_readout_only; do
  for s in $SEEDS; do
    src="$RUNS/$MT/seed$s/model.pt"
    if [ ! -f "$src" ]; then
      n_missing=$((n_missing + 1))
      missing_list+=("$MT/seed$s")
      continue
    fi
    n_found=$((n_found + 1))
    echo ">> reversal $MT seed$s  (n_trials=$N_TRIALS)"
    "$PY" "$HERE/train_reversal.py" --run "$RUNS/$MT/seed$s" --out "$OUT/$MT/seed$s" \
        --n-trials "$N_TRIALS" --checkpoint-every "$CHECKPOINT_EVERY" \
        --checkpoint-fine-until "$CHECKPOINT_FINE_UNTIL" --checkpoint-fine-every "$CHECKPOINT_FINE_EVERY"
    n_done=$((n_done + 1))
  done
done

if [ "$n_missing" -gt 0 ]; then
  echo "WARNING: $n_missing seed(s) had no source model.pt under $RUNS and were skipped:" >&2
  printf '  - %s\n' "${missing_list[@]}" >&2
fi

if [ "$n_found" -eq 0 ]; then
  echo "ERROR: found 0 source models under RUNS=$RUNS for any (model_type, seed) -- nothing was trained." >&2
  echo "       Check that RUNS points at a directory containing <model_type>/seed<NN>/model.pt," >&2
  echo "       e.g. RUNS=$HERE/../../model_runs (this run's default)." >&2
  exit 1
fi

if [ "$n_done" -ne "$n_found" ]; then
  # Shouldn't happen given `set -e`, but guard anyway in case that ever changes.
  echo "ERROR: only trained $n_done of $n_found found runs -- something failed partway." >&2
  exit 1
fi

echo "done. Trained $n_done reversal run(s) into $OUT (skipped $n_missing missing seed(s))."
echo "Next: python \"$HERE/../inference/run_inference.py\" --runs \"$OUT\" --out figure_data_reversal ; then figures.py"
