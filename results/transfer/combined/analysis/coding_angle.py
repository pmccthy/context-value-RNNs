"""coding_angle.py

Average normalised per-condition coding vectors and their pairwise cosine
similarity ("coding angle"), computed per model_type/seed from figure_data.pkl,
then averaged across seeds. Lets us ask e.g. whether 0%-vs-100% coding
directions are more/less separated in one model class than another.

The post-reversal figure_data (figure_data_reversal[REV_TAG]) is loaded the
same way so pre- vs post-reversal geometry can be compared for the RNNs, in
the same structure used on the experimental side (see coding_angle_experimental.py
in neuronal-representations, once written -- pre/post-Rev x 3 stimulus stems).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "transfer" / "code"))
sys.path.insert(0, str(_HERE.parent / "panels"))  # for vigour_value.recovered_seeds

import figures as F  # noqa: E402
import reversal_gradients as RG  # noqa: E402
import vigour_value as VV  # noqa: E402

REV_TAG = RG.REV_TAG
FIGURE_DATA_DIR = _HERE.parent / "transfer" / "figure_data"
FIGURE_DATA_REV_DIR = _HERE.parent / "transfer" / f"figure_data_reversal{REV_TAG}"

_D_CACHE: dict = {}


def _D(phase="pre"):
    if phase not in _D_CACHE:
        path = FIGURE_DATA_DIR if phase == "pre" else FIGURE_DATA_REV_DIR
        _D_CACHE[phase] = F.load(str(path))
    return _D_CACHE[phase]


def _default_seeds(model_type: str, phase: str):
    """Recovered-only seeds (vigour_value.recovered_seeds), intersected with
    seeds present in this phase's figure_data -- falling back to every
    present seed if recovery filtering finds none. See
    circuit_graph._default_seeds for the identical convention applied there,
    and vigour_value.py's module docstring for why recovered-only is the
    right default even for pre-reversal summaries."""
    present = set(_D(phase)["seeds"])
    rec = VV.recovered_seeds(model_type)
    if rec:
        kept = sorted(rec & present)
        if kept:
            return kept
    return sorted(present)


def condition_coding_vectors(model_type: str, seed: int, phase: str = "pre") -> np.ndarray | None:
    """Average-normalised per-condition coding vector, one row per stimulus condition.

    Returns (n_stim, n_units) with each row L2-normalised, or None if this
    seed has no finite data for this model_type/phase.
    """
    D = _D(phase)
    ti = D["model_types"].index(model_type)
    si = D["seeds"].index(seed)
    arr = D["aligned_mean"]["data"][ti, si]  # (unit, time, stim)
    if not np.isfinite(arr).all():
        return None
    period = D["period"]
    stim_epoch = slice(period["n_iti_pre"], period["n_iti_pre"] + period["stim_ts"])
    vecs = arr[:, stim_epoch, :].mean(axis=1).T  # (n_stim, n_units)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return vecs / norms


def pairwise_cosine(vecs: np.ndarray) -> np.ndarray:
    """vecs: (n_stim, n_units), already unit-normalised -> (n_stim, n_stim) cosine similarity."""
    return vecs @ vecs.T


def coding_angle_summary(model_type: str, phase: str = "pre", seeds=None):
    """Mean +/- SEM cosine-similarity matrix across seeds, and stim labels.

    Returns (mean_mat (n_stim,n_stim), sem_mat (n_stim,n_stim), n_seeds, stim_labels).
    """
    D = _D(phase)
    seeds = seeds if seeds is not None else _default_seeds(model_type, phase)
    mats = []
    for seed in seeds:
        vecs = condition_coding_vectors(model_type, seed, phase=phase)
        if vecs is None:
            continue
        mats.append(pairwise_cosine(vecs))
    if not mats:
        raise FileNotFoundError(f"no coding-vector data for {model_type}/{phase}")
    mats = np.asarray(mats)
    mean = mats.mean(axis=0)
    sem = mats.std(axis=0, ddof=1) / np.sqrt(mats.shape[0]) if mats.shape[0] > 1 else np.zeros_like(mean)
    return mean, sem, mats.shape[0], D["stim_labels"]


# Physical-stimulus-index transition labels, matching circuit_graph's
# STIM_HEAD_ROW_LABELS convention: D["stim_labels"] (["0%","50%","100%"]) are
# phase-blind physical-stimulus-INDEX labels -- the same stimulus index's
# actual reward association flips at the reversal (VM_PRE=[0,.5,1] ->
# VM_POST=[1,.5,0], 50% stim unchanged), so a bare "0%"/"100%" would silently
# mean opposite things depending on phase. Relabelled here the same way the
# circuit diagrams are.
_TRANSITION_LABEL = {"0%": "0%→100%", "100%": "100%→0%"}


def combined_condition_coding_vectors(model_type: str, seed: int) -> np.ndarray | None:
    """Concatenate this seed's pre- and post-reversal per-condition coding
    vectors into one (2*n_stim, n_units) array -- pre-reversal stimulus rows
    first, then post-reversal. Safe to concatenate directly (not just stack
    for display): unit ordering is identical between figure_data.pkl (pre)
    and figure_data_reversal{REV_TAG}.pkl (post) for a given seed --
    reversal_analysis.py only ever transposes these arrays, never reorders
    units, so row i in both blocks refers to the same physical hidden unit.
    Returns None if either phase is missing/non-finite for this seed."""
    vp = condition_coding_vectors(model_type, seed, phase="pre")
    vq = condition_coding_vectors(model_type, seed, phase="post")
    if vp is None or vq is None:
        return None
    return np.concatenate([vp, vq], axis=0)


def combined_coding_angle_summary(model_type: str, seeds=None):
    """Mean +/- SEM cosine-similarity matrix across seeds for the COMBINED
    pre+post condition set (2*n_stim x 2*n_stim: 3 pre-reversal stimulus
    conditions then 3 post-reversal ones), and combined transition-aware
    labels. This matrix is exactly symmetric with an exactly-1.0 diagonal by
    construction (it's vecs @ vecs.T on unit-normalised rows), so a caller
    that wants a triangular display can safely mask/drop the redundant half.

    Seeds default to the recovered-only set (vigour_value.recovered_seeds),
    which for the combined matrix additionally requires the seed to have
    usable data in BOTH phases (a seed present pre-reversal but missing
    dense post-reversal figure_data can't contribute a combined row, even if
    it's nominally "recovered").

    Returns (mean_mat (2n,2n), sem_mat (2n,2n), n_seeds, labels).
    """
    seeds = seeds if seeds is not None else _default_seeds(model_type, "post")
    mats = []
    for seed in seeds:
        vecs = combined_condition_coding_vectors(model_type, seed)
        if vecs is None:
            continue
        mats.append(pairwise_cosine(vecs))
    if not mats:
        raise FileNotFoundError(f"no combined pre+post coding-vector data for {model_type}")
    mats = np.asarray(mats)
    mean = mats.mean(axis=0)
    sem = mats.std(axis=0, ddof=1) / np.sqrt(mats.shape[0]) if mats.shape[0] > 1 else np.zeros_like(mean)
    stim = _D("pre")["stim_labels"]
    trans_labels = [_TRANSITION_LABEL.get(s, s) for s in stim]
    labels = [f"{s} (pre)" for s in trans_labels] + [f"{s} (post)" for s in trans_labels]
    return mean, sem, mats.shape[0], labels
