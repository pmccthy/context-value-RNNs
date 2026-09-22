"""Shared responder-method toggle for the model-side panel/compose modules,
mirroring the neural-data repo's panels/subgroups.py SG.METHOD pattern.

RESPONDER_METHOD (env var, or set the module attribute directly before
calling a build_all()/figure function): "time_averaged" (default -- the
window-mean paired t-test that has always populated figure_data.pkl's
"responsive" field) or "temporal" (the temporal-cluster t-test, ported
verbatim from the real-data method -- see model_responders.py).

load_D(F, base_dir, method=None) loads figure_data.pkl from `base_dir` and,
if method == "temporal", swaps in model_responders.build_temporal_D() using
base_dir/time_resolved. Raises FileNotFoundError (uncaught) when the
temporal method's raw data isn't there yet -- callers should catch it and
report/skip rather than silently fall back, since a silent fallback would
produce a "temporal" figure that's secretly time-averaged data.
"""
from __future__ import annotations

import os
from pathlib import Path

METHOD = os.environ.get("RESPONDER_METHOD", "time_averaged")
assert METHOD in ("time_averaged", "temporal"), f"bad RESPONDER_METHOD: {METHOD!r}"

METHOD_LABEL = {"time_averaged": "time-averaged (window-mean) t-test",
                "temporal": "temporal-cluster t-test"}
METHOD_FOLDER = {"time_averaged": "time_averaged", "temporal": "temporal"}


_CACHE = {}  # (str(base_dir), method) -> D ; temporal build is ~20s/90 files, worth caching
              # across the several build_*() calls each panel module makes per run.


def load_D(F, MR, base_dir, method=None):
    method = method or METHOD
    key = (str(base_dir), method)
    if key in _CACHE:
        return _CACHE[key]
    D = F.load(str(base_dir))
    if method == "temporal":
        D = MR.build_temporal_D(D, str(Path(base_dir) / "time_resolved"))
    _CACHE[key] = D
    return D
