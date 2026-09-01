#!/usr/bin/env bash
# Train all three model arms x seeds for the final 3s1c vigour study.
# Each run writes model.pt, model_init.pt, config.json, meta.txt under
#   <OUT>/<model_type>/seed<NN>/   (default OUT = ../../model_runs).
# Usage:  [SEEDS="42 43 44"] [OUT=/path] [PY=python] bash run_all.sh
set -euo pipefail
PY="${PY:-python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OUT:-$HERE/../../model_runs}"
SEEDS="${SEEDS:-$(seq 42 71)}"
PROBE_EVERY="${PROBE_EVERY:-20}"     # log per-stim metrics during training (0 to disable);
                                     # needed for the combined pre->post reversal timeline
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-0}"   # 0 = off (default); >0 saves a full model
                                             # checkpoint every N updates under
                                             # <out>/checkpoints/ (plus always update 1) --
                                             # needed by compute_pretrain_crosscontext_decode.py
                                             # to build a genuine trials-resolved pre-reversal
                                             # cross-context decode trajectory after the fact.
CHECKPOINT_FINE_UNTIL="${CHECKPOINT_FINE_UNTIL:-$CHECKPOINT_EVERY}"  # fine-resolution window
                                             # at the start of training (default: fills the
                                             # first coarse interval)
CHECKPOINT_FINE_EVERY="${CHECKPOINT_FINE_EVERY:-1}"   # fine-window interval (default: every update)
for MT in classif_rl rl_only classif_rl_readout_only; do
  for s in $SEEDS; do
    echo ">> $MT seed$s"
    "$PY" "$HERE/train_model.py" --model-type "$MT" --seed "$s" --out "$OUT/$MT/seed$s" \
        --probe-every "$PROBE_EVERY" --checkpoint-every "$CHECKPOINT_EVERY" \
        --checkpoint-fine-until "$CHECKPOINT_FINE_UNTIL" \
        --checkpoint-fine-every "$CHECKPOINT_FINE_EVERY"
  done
done
echo "done. Next: python ../inference/run_inference.py --runs $OUT --out ../../figure_data"
