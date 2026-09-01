#!/usr/bin/env python
"""Run inference on every trained model and write all the data the figures need.

ONE deterministic inference pass per model (default 8 episodes x 200 trials =
1600 interleaved trials; ~530 per stimulus) produces, per model:

  * per-trial, time-resolved, trial-aligned hidden activity  (the PSTH source)
  * per-unit responsiveness pattern (paired t-test vs ITI baseline) -> cell groups
  * per-unit stimulus tuning (stim-window mean)                      -> heatmaps
  * scalar metrics: vigour, population activity, responder counts/fractions -> bars

Outputs (under --out)
---------------------
  figure_data.pkl   self-documenting dict (dims + coords on every array): scalars,
                    per-unit tuning, per-unit trial-aligned means, responsiveness.
                    This single file backs ALL the figures.
  metrics.csv       the scalar metrics in tidy long form (model_type, seed, stim).
  activations.npz   per-unit stim-window tuning as (3, H) arrays keyed "{type}_{seed}".
  time_resolved/<type>_seed<NN>.npz   the raw per-trial trial-aligned hidden:
                    aligned (n_trials, n_align_ts, H), stimulus (n_trials,),
                    vigour (n_trials,), bounds, segment labels.

The pass is resumable (per-model cache under <out>/_cache); pass --budget SECONDS
to stop computing new models after that wall-clock time and re-run to continue.

Usage:
  python scripts/16_06_26_run_inference.py --runs results/16_06_26_final/model_runs \
         --out results/16_06_26_final/figure_data
"""
from __future__ import annotations
import argparse, glob, json, pickle, re, sys, time
from pathlib import Path
import numpy as np, torch, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cxval.models import RNN
from cxval.batched import generate_batch
from cxval.vigour import VigourActorCritic, BatchedVigourEnv
from cxval.analysis import responsive_proportions_ttest

VM = np.array([[0.0], [0.5], [1.0]], dtype=np.float32)
STIM_LABELS = ["0%", "50%", "100%"]
SEGMENTS = ["ITI", "stim", "outcome"]
# --- inference settings (the one knob for trial count; stim stays interleaved) ---
N_EVAL_EPISODES = 8
N_TRIALS_PER_EPISODE = 200            # -> 1600 trials/model, ~530 per stimulus
N_ITI_PRE = 3                         # pre-stimulus baseline steps kept in the aligned window
EVAL_SEED = 10_000
COST = 0.8
METRIC_DESC = {
    "vigour":          "deterministic mean vigour per stimulus, in [0,1]",
    "pop_activity":    "mean hidden-unit activation in the stim window",
    "n_responsive":    "# units significantly responsive (paired t-test vs pre-stim baseline)",
    "frac_responsive": "fraction of units significantly responsive",
}


def build_model(state_dict, min_vigour=0.0, squash_width=None):
    """Reconstruct a VigourActorCritic from a saved state_dict (architecture inferred
    from weight shapes; aux classification head included iff present). `min_vigour`/
    `squash_width` MUST match the run's actual training config (its squash() shape) --
    inference here is deterministic (squash(mean)), so a mismatch would silently give
    wrong vigour readings for any run trained with a nonzero floor or soft-clamp width."""
    H = state_dict["backbone.h2h.weight"].shape[0]
    obs = state_dict["backbone.input2h.weight"].shape[1]
    rf = state_dict["vigour_head.weight"].shape[1] / H
    aux = state_dict["stim_head.weight"].shape[0] if "stim_head.weight" in state_dict else 0
    ac = VigourActorCritic(RNN(input_size=obs, hidden_size=H, output_size=1,
                               recurrent_gain=0.9), action_std=0.05,
                           readout_fraction=rf, aux_n_stim=aux)
    ac.min_vigour = min_vigour
    ac.squash_width = squash_width
    ac.load_state_dict(state_dict); ac.eval()
    return ac


@torch.no_grad()
def run_inference(model, *, n_eval_episodes=N_EVAL_EPISODES,
                  n_trials_per_episode=N_TRIALS_PER_EPISODE, base_seed=EVAL_SEED,
                  vigour_cost=COST, n_iti_pre=N_ITI_PRE):
    """Deterministic eval of one model. Returns a dict with the per-trial aligned
    activity and everything derived from it (see module docstring). The trial-aligned
    window is [ITI_pre | stim | outcome] so axis-1 is comparable across models."""
    sb, rbv, abv, structs = generate_batch(VM, n_trials_per_episode, n_eval_episodes, base_seed)
    env = BatchedVigourEnv(sb, rbv, abv, vigour_cost=vigour_cost)
    B, T, _ = sb.shape
    H = model.backbone.hidden_size

    # roll out: record the full hidden stream + the deterministic vigour
    obs = torch.as_tensor(env.reset(), dtype=torch.float32)
    hidden = None
    hid = np.zeros((B, T, H), np.float32)
    vig = np.zeros((B, T), np.float32)
    done, t = False, 0
    while not done:
        mean, _, hidden = model.step(obs, hidden)
        v = model.squash(mean)
        hid[:, t] = hidden.cpu().numpy()
        vig[:, t] = v.cpu().numpy()
        o, _, done, _ = env.step(v.cpu().numpy())
        obs = torch.as_tensor(o, dtype=torch.float32)
        t += 1

    stim_ts = structs[0][0]["stim_window"][1] - structs[0][0]["stim_window"][0]
    rew_ts = structs[0][0]["reward_window"][1] - structs[0][0]["reward_window"][0]
    n_align = n_iti_pre + stim_ts + rew_ts

    # slice each trial into the aligned [ITI|stim|outcome] window
    aligned, labels, vig_trial = [], [], []
    for b, struct in enumerate(structs):
        for tr in struct:
            ss, se = tr["stim_window"]; rs, re = tr["reward_window"]
            if ss - n_iti_pre < 0:
                continue
            win = np.concatenate([hid[b, ss - n_iti_pre:ss], hid[b, ss:se], hid[b, rs:re]], 0)
            aligned.append(win)                                  # (n_align, H)
            labels.append(tr["stimulus"])
            vig_trial.append(float(vig[b, rs:re].mean()))        # reward-window mean vigour
    aligned = np.stack(aligned).astype(np.float32)               # (n_trials, n_align, H)
    labels = np.asarray(labels)
    vig_trial = np.asarray(vig_trial, np.float32)

    # derived views (all from `aligned`, so slices stay consistent)
    sl_base = slice(0, n_iti_pre)
    sl_stim = slice(n_iti_pre, n_iti_pre + stim_ts)
    stim_hidden = aligned[:, sl_stim]                            # (n_trials, stim_ts, H)
    base_hidden = aligned[:, sl_base]                            # (n_trials, n_iti_pre, H)
    acts = {"stim_hidden": stim_hidden, "baseline_hidden": base_hidden, "stimulus": labels}
    rp = responsive_proportions_ttest(acts, period="stim")

    R = stim_hidden.mean(1)                                      # (n_trials, H) per-unit stim mean
    tuning = np.stack([R[labels == s].mean(0) for s in range(3)])            # (3, H)
    aligned_mean = np.stack([aligned[labels == s].mean(0) for s in range(3)])  # (3, n_align, H)

    return {
        "scalars": {
            "vigour":          np.array([vig_trial[labels == s].mean() for s in range(3)], float),
            "pop_activity":    tuning.mean(1).astype(float),
            "n_responsive":    np.asarray(rp["n_responsive"], float),
            "frac_responsive": np.asarray(rp["frac_per_stim"], float),
        },
        "tuning":        tuning.astype(np.float32),              # (3, H)
        "aligned_mean":  aligned_mean.astype(np.float32),        # (3, n_align, H)
        "responsive":    np.asarray(rp["responsive"], bool),     # (H, 3)
        "bounds":        (n_iti_pre, int(stim_ts), int(rew_ts)),
        "raw": {"aligned": aligned, "stimulus": labels, "vigour": vig_trial},
    }


def discover(runs_root):
    """Find model runs as <runs_root>/<model_type>/seed<NN>/model.pt."""
    jobs = []
    for type_dir in sorted(glob.glob(str(Path(runs_root) / "*"))):
        if not Path(type_dir).is_dir():
            continue
        mtype = Path(type_dir).name
        for d in sorted(glob.glob(str(Path(type_dir) / "seed*")),
                        key=lambda p: int(re.search(r"seed(\d+)", p).group(1))):
            if (Path(d) / "model.pt").exists():
                jobs.append((mtype, int(re.search(r"seed(\d+)", d).group(1)), Path(d)))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/16_06_26_final/model_runs")
    ap.add_argument("--out", default="results/16_06_26_final/figure_data")
    ap.add_argument("--budget", type=float, default=0.0,
                    help="stop computing NEW models after this many seconds (0=no limit); "
                         "results cached so re-running continues.")
    args = ap.parse_args()
    out = Path(args.out); (out / "_cache").mkdir(parents=True, exist_ok=True)
    (out / "time_resolved").mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    jobs = discover(args.runs)
    if not jobs:
        print(f"no runs found under {args.runs}"); return
    results, n_new, n_pending = {}, 0, 0
    for mtype, seed, d in jobs:
        cf = out / "_cache" / f"{mtype}_{seed}.pkl"
        if cf.exists():
            r = pickle.load(open(cf, "rb"))
        else:
            if args.budget and (time.time() - t0) > args.budget:
                n_pending += 1; continue
            sd = torch.load(d / "model.pt", map_location="cpu")
            if any(not torch.isfinite(v).all() for v in sd.values()):
                r = None
            else:
                cf_json = d / "config.json"
                mv, sw = 0.0, None
                if cf_json.exists():
                    tr_cfg = json.loads(cf_json.read_text()).get("train", {})
                    mv = tr_cfg.get("min_vigour", 0.0)
                    sw = tr_cfg.get("squash_width")
                r = run_inference(build_model(sd, min_vigour=mv, squash_width=sw))
                # write the raw per-trial time-resolved file immediately (resumable)
                raw = r.pop("raw")
                np.savez_compressed(
                    out / "time_resolved" / f"{mtype}_seed{seed}.npz",
                    aligned=raw["aligned"], stimulus=raw["stimulus"], vigour=raw["vigour"],
                    bounds=np.array(r["bounds"]), segment_labels=np.array(SEGMENTS),
                    stim_labels=np.array(STIM_LABELS),
                    meta=np.array(json.dumps({"model_type": mtype, "seed": seed,
                                              "dims_aligned": ["trial", "time", "unit"]})))
            pickle.dump(r, open(cf, "wb")); n_new += 1
            print(f"  inferred {mtype} seed{seed}" + ("  (DIVERGED)" if r is None else ""))
        if r is not None:
            results[(mtype, seed)] = r

    n_done = sum((out / "_cache" / f"{m}_{s}.pkl").exists() for m, s, _ in jobs)
    if n_done < len(jobs):
        print(f"\n{n_done}/{len(jobs)} models done (+{n_new} this run, {n_pending} pending). Re-run to continue.")
        return

    _assemble(results, jobs, out)


def _assemble(results, jobs, out):
    """Pack the per-model results into the labelled figure_data.pkl + metrics.csv +
    activations.npz. Seeds can differ across model types; the dense arrays use the
    union of seeds and are NaN where a (type, seed) is absent."""
    types = sorted({m for m, _, _ in jobs}, key=_type_order)
    seeds = sorted({s for _, s, _ in jobs})
    ti = {m: i for i, m in enumerate(types)}; si = {s: i for i, s in enumerate(seeds)}
    A, S, K = len(types), len(seeds), 3
    H = next(iter(results.values()))["tuning"].shape[1]
    n_align = next(iter(results.values()))["aligned_mean"].shape[1]
    bounds = next(iter(results.values()))["bounds"]

    scal = {m: np.full((A, S, K), np.nan) for m in METRIC_DESC}
    tuning = np.full((A, S, K, H), np.nan, np.float32)
    aligned = np.full((A, S, H, n_align, K), np.nan, np.float32)     # (type,seed,unit,time,stim)
    responsive = np.zeros((A, S, H, K), bool)
    rows = []
    for (mtype, seed), r in results.items():
        a, s = ti[mtype], si[seed]
        for m in METRIC_DESC:
            scal[m][a, s] = r["scalars"][m]
        tuning[a, s] = r["tuning"]
        aligned[a, s] = np.transpose(r["aligned_mean"], (2, 1, 0))   # (unit,time,stim)
        responsive[a, s] = r["responsive"]
        for k in range(3):
            rows.append(dict(model_type=mtype, seed=seed, stim=k, stim_label=STIM_LABELS[k],
                             vigour=r["scalars"]["vigour"][k],
                             pop_activity=r["scalars"]["pop_activity"][k],
                             n_responsive=int(r["scalars"]["n_responsive"][k]),
                             frac_responsive=r["scalars"]["frac_responsive"][k], n_units=H))

    coords = {"model_type": types, "seed": seeds, "stim": STIM_LABELS}
    n_iti_pre, stim_ts, rew_ts = bounds
    data = {
        "description": "Final 3s1c continuous-vigour models: per-(model_type, seed, stimulus) "
                       "metrics, per-unit stim tuning, per-unit trial-aligned means, and "
                       "responsiveness. Backs every figure. Raw per-trial activity is in "
                       "time_resolved/.",
        "model_types": types, "seeds": seeds, "stim_labels": STIM_LABELS, "n_units": H,
        "seeds_present": {m: sorted({s for mm, s, _ in jobs if mm == m}) for m in types},
        "period": {"n_iti_pre": n_iti_pre, "stim_ts": stim_ts, "rew_ts": rew_ts,
                   "n_align_ts": n_align, "segments": SEGMENTS,
                   "segment_bounds": [[0, n_iti_pre], [n_iti_pre, n_iti_pre + stim_ts],
                                      [n_iti_pre + stim_ts, n_align]]},
        "scalars": {"dims": ["model_type", "seed", "stim"], "coords": coords,
                    "units": METRIC_DESC, **scal},
        "tuning": {"description": "per-unit mean stim-window activation",
                   "dims": ["model_type", "seed", "stim", "unit"],
                   "coords": {**coords, "unit": list(range(H))}, "data": tuning},
        "aligned_mean": {"description": "per-unit trial-aligned mean activation per stimulus "
                                        "(time axis = ITI|stim|outcome)",
                         "dims": ["model_type", "seed", "unit", "time", "stim"],
                         "coords": {"model_type": types, "seed": seeds,
                                    "unit": list(range(H)), "time": list(range(n_align)),
                                    "stim": STIM_LABELS}, "data": aligned},
        "responsive": {"description": "unit significantly responds to stimulus (defines groups)",
                       "dims": ["model_type", "seed", "unit", "stim"],
                       "coords": {"model_type": types, "seed": seeds,
                                  "unit": list(range(H)), "stim": STIM_LABELS},
                       "data": responsive},
    }
    pickle.dump(data, open(out / "figure_data.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    npz = {f"{m}_{s}": r["tuning"] for (m, s), r in results.items()}
    np.savez(out / "activations.npz", **npz)

    print(f"\n{len(results)} models | types={types} | seeds {seeds[0]}..{seeds[-1]} | H={H} "
          f"| aligned window={n_align} ({'+'.join(map(str, bounds))})")
    print(f"  -> {out/'figure_data.pkl'}")
    print(f"  -> {out/'metrics.csv'}  ({len(rows)} rows)")
    print(f"  -> {out/'activations.npz'}  ({len(npz)} keys)")
    print(f"  -> {out/'time_resolved'}/  (per-trial raw, one .npz per model)")


def _type_order(m):
    order = {"classif_rl": 0, "rl_only": 1, "classif_rl_readout_only": 2}
    return (order.get(m, 9), m)


if __name__ == "__main__":
    main()
