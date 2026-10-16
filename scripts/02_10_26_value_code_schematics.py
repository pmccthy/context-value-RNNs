"""
Hypothetical codes an RNN critic could use to represent both stimulus identity and
stimulus value -- built from an actual (tiny) sampled network.

Generative model (one NetCode = one hypothetical solution)
----------------------------------------------------------
* N hidden units are split into G groups (G = 3 subgroups, or G = 1 single population)
  in proportion to `sizes`.
* For a unit i in group g:
      input weight from stimulus s   w_in[i, s] ~ Normal(mu_in[g, s], sd_in)
      readout weight                 w_out[i]   ~ Normal(mu_out[g],   sd_out)
      bias (threshold)               b[i]       ~ Normal(mu_b,        sd_b)
  Recurrent weights exist only WITHIN a group:
      W_rec[i, j] ~ Normal(mu_rec[g] / n_g, sd_rec / sqrt(n_g))  for i, j in group g
  (so each row sums to ~mu_rec[g], i.e. mu_rec is the group's recurrent gain).
* Activity to stimulus s (one-hot input) is the steady state of a ReLU rate network:
      r = ReLU(w_in[:, s] + W_rec r + b)
* Value output is an actual readout of the units:
      V_hat(s) = w_out . r(s) / N
  With readout="fit" the readout weights are instead learned: the minimum-norm w_out
  that maps r(s) exactly onto the values (plus an output bias) -- "readout learning"
  on whatever features the network has.

Figures
-------
Schematic (group-level MEANS of the sampled parameters):
  node AREA ~ number of units in the group; node SHADING ~ the group's mean activity
  to its own stimulus (single population: horizontal bands = population mean
  activity under each stimulus); edge WIDTH ~ |mean weight|; self-loop ~ mu_rec.
Heatmap: the simulated stimulus x unit activity, sorted exactly as transfer heatmap_seed
  (by preferred stimulus, then peak; units with peak < 5% of max at the bottom).
explore(): adds the actual V_hat per stimulus and the responder categories
  (which stimuli each unit responds to, threshold RESP_THRESH) as in transfer.

    python scripts/02_10_26_value_code_schematics.py      # regenerate saved figures
Interactive: nb/02_10_26_value_code_schematics_explorer.ipynb
Author: patrick.mccarthy@dpag.ox.ac.uk
"""
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

# Stimulus colours/labels as in the transfer figures (results/transfer/code/figure_config.py)
STIM_COLORS = ["#fab079", "#b8397a", "#2E215E"]   # 0% / 50% / 100%

# --------------------------------------------------------------------------
# Global knobs (read at CALL time, so changing them on the imported module works)
# --------------------------------------------------------------------------
VALUES = np.array([0.0, 0.5, 1.0])   # stimulus values (task: 0% / 50% / 100%)
STIM_NAMES = ["0%", "50%", "100%"]
BASELINE = 0.25   # floor so a 0-value stimulus is still represented (used by the presets)
OUT_DIR = Path(__file__).resolve().parents[1] / "figures" / "02_10_26_value_code_schematics"

N_UNITS = 60             # hidden units (divisible by 3 so equal groups get equal counts)
SEED = 0
N_STEPS = 300            # fixed-point iterations for the recurrent steady state
RESP_THRESH = 0.1        # a unit "responds" to a stimulus if its activity exceeds this
HEAT_VMAX = 1.2          # shared heatmap colour ceiling
LEVEL_VMAX = 1.0         # activity that maps to the darkest grey in the schematic

LEVEL_CMAP = LinearSegmentedColormap.from_list("level", ["#ffffff", "#262626"])  # greyscale
HEAT_CMAP = "hot"
MAX_RADIUS = 0.55        # radius of the single-population node
GROUP_MAX_RADIUS = 0.40  # radius of the largest subgroup
MAX_LW = 7.0             # linewidth of the largest weight in the figure
MIN_LW = 0.6
EDGE_MIN = 0.03          # input edges with |mean weight| below this are not drawn
FEATURE_COLS = ["Subgroup structure", "Value-proportionate subgroups",
                "Structured readout weights", "Level-based coding"]
SHOW_FEATURES = True
SHOW_HEATMAPS = True

# responder categories, in the order used by transfer's responder_groups figure
RESP_CATS = [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
RESP_LABELS = ["0%\nonly", "50%\nonly", "100%\nonly", "0%\n&50%", "0%\n&100%", "50%\n&100%", "all\nthree"]


def affine(v, b=None):
    """Map values in [0,1] to [b,1] (a floor so the 0% stimulus is still represented)."""
    b = BASELINE if b is None else b
    return b + (1 - b) * np.asarray(v, float)


# --------------------------------------------------------------------------
# The generative model
# --------------------------------------------------------------------------
@dataclass
class NetCode:
    name: str
    note: str = ""
    sizes: np.ndarray = field(default_factory=lambda: np.ones(3))      # (G,) relative group sizes
    mu_in: np.ndarray = field(default_factory=lambda: np.eye(3))       # (G, 3) mean input weight
    sd_in: float = 0.1
    mu_out: np.ndarray = field(default_factory=lambda: np.ones(3))     # (G,) mean readout weight
    sd_out: float = 0.1
    mu_rec: np.ndarray = field(default_factory=lambda: np.zeros(3))    # (G,) within-group recurrent gain
    sd_rec: float = 0.0
    mu_b: float = 0.0
    sd_b: float = 0.0
    readout: str = "sampled"   # "sampled" or "fit" (learned min-norm readout)
    n_units: int = None        # None -> N_UNITS
    seed: int = None           # None -> SEED

    @property
    def G(self):
        return len(np.atleast_1d(self.sizes))

    def _arr(self):
        """Parameters as arrays of the right shape. Longer-than-needed vectors are truncated
        to the first G entries (so a single population can reuse 3-entry defaults)."""
        G = self.G
        def vec(x):
            x = np.atleast_1d(np.asarray(x, float))
            return np.full(G, x[0]) if len(x) == 1 else x[:G]
        mu_in = np.atleast_2d(np.asarray(self.mu_in, float))[:G]
        if mu_in.shape[0] == 1 and G > 1:
            mu_in = np.repeat(mu_in, G, 0)
        return mu_in, vec(self.mu_out), vec(self.mu_rec)

    def sample(self):
        """Draw one network. Returns dict(group, W_in (N,3), W_rec (N,N), b (N,), w_out (N,))."""
        N = N_UNITS if self.n_units is None else self.n_units
        rng = np.random.default_rng(SEED if self.seed is None else self.seed)
        mu_in, mu_out, mu_rec = self._arr()
        sizes = np.clip(np.atleast_1d(np.asarray(self.sizes, float)), 0, None)
        exact = sizes / sizes.sum() * N            # largest-remainder allocation of N units
        counts = np.floor(exact).astype(int)
        for k in np.argsort(-(exact - counts), kind="stable")[:N - counts.sum()]:
            counts[k] += 1
        group = np.repeat(np.arange(self.G), counts)
        W_in = rng.normal(mu_in[group], self.sd_in)                    # (N, 3)
        w_out = rng.normal(mu_out[group], self.sd_out)                 # (N,)
        b = rng.normal(self.mu_b, self.sd_b, N)
        W_rec = np.zeros((N, N))
        for g in range(self.G):
            idx = np.where(group == g)[0]
            n = len(idx)
            if n and (mu_rec[g] != 0 or self.sd_rec > 0):
                W_rec[np.ix_(idx, idx)] = rng.normal(mu_rec[g] / n, self.sd_rec / np.sqrt(n), (n, n))
        return dict(group=group, W_in=W_in, W_rec=W_rec, b=b, w_out=w_out, counts=counts)

    def simulate(self):
        """Sample the network and run it. Returns dict with
        R (3, N) steady-state activity, vhat (3,) readout, net (the sampled weights),
        level (per group own-stimulus mean, or per stimulus for a single population)."""
        net = self.sample()
        N = len(net["b"])
        if N and np.max(np.abs(np.linalg.eigvals(net["W_rec"]))) >= 1:
            print(f"WARNING '{self.name}': recurrent spectral radius >= 1, steady state may not exist")
        drive = net["W_in"].T + net["b"]                # (3, N) feedforward drive per stimulus
        if not np.any(net["W_rec"]):
            R = np.maximum(drive, 0)
        else:                                          # damped fixed-point iteration, all stimuli at once
            R = np.zeros((3, N))
            for _ in range(N_STEPS):
                R = 0.5 * R + 0.5 * np.maximum(drive + R @ net["W_rec"].T, 0)
        if self.readout == "fit":               # learned readout: min-norm exact fit (+ bias)
            Rc = R - R.mean(0)
            w = np.linalg.pinv(Rc) @ (VALUES - VALUES.mean())
            net["w_out"] = w * N                # so that vhat = R @ w_out / N
        vhat = R @ net["w_out"] / max(N, 1)
        g = net["group"]
        if self.G == 3:
            level = np.array([R[k, g == k].mean() if np.any(g == k) else 0.0 for k in range(3)])
        else:
            level = R.mean(1)
        return dict(R=R, vhat=vhat, net=net, level=level)


def readout_fit(vhat):
    """How well an output gain + bias maps V_hat onto VALUES. Returns (fitted, r2, slope)."""
    A = np.c_[vhat, np.ones(3)]
    coef, *_ = np.linalg.lstsq(A, VALUES, rcond=None)
    fitted = A @ coef
    ss = np.sum((VALUES - VALUES.mean()) ** 2)
    return fitted, 1 - np.sum((VALUES - fitted) ** 2) / ss, coef[0]


def _codes_value(x, rel_tol=0.25):
    """x varies across the 3 stimuli by > rel_tol of its mean and increases with value."""
    x = np.broadcast_to(np.asarray(x, float), 3)
    spread_ok = np.ptp(x) > rel_tol * max(np.abs(x).mean(), 1e-9)
    return bool(spread_ok and np.corrcoef(x, VALUES)[0, 1] > 0)


def infer_features(code, sim=None):
    """The 4 yes/no features, read off the parameters and the simulated activity."""
    sim = code.simulate() if sim is None else sim
    mu_in, mu_out, _ = code._arr()
    yn = lambda b: "Y" if b else "N"
    sub = code.G > 1
    return (yn(sub),
            yn(_codes_value(code.sizes)) if sub else "N/A",
            yn(code.readout == "fit" or (sub and _codes_value(mu_out))),
            yn(_codes_value(sim["level"])))


def responder_counts(R, thresh=None):
    """Number of units in each responder category (RESP_CATS order) + 'none'."""
    thresh = RESP_THRESH if thresh is None else thresh
    pats = [tuple(np.where(R[:, i] > thresh)[0]) for i in range(R.shape[1])]
    return np.array([sum(p == c for p in pats) for c in RESP_CATS] + [sum(len(p) == 0 for p in pats)])


def independence_check(R, thresh=None):
    """Compare responder-category counts with what you'd expect if each unit responded to
    each stimulus INDEPENDENTLY (product of the per-stimulus response probabilities).
    Random mixed selectivity ~ matches independence; true subgroups show an excess of
    single-stimulus responders and a deficit of multi-responders.
    Returns (observed, expected) over RESP_CATS + ['none']."""
    thresh = RESP_THRESH if thresh is None else thresh
    resp = R > thresh                                   # (3, N)
    p = resp.mean(1)
    N = R.shape[1]
    exp = []
    for c in RESP_CATS + [()]:
        exp.append(N * np.prod([p[s] if s in c else 1 - p[s] for s in range(3)]))
    return responder_counts(R, thresh), np.array(exp)


# --------------------------------------------------------------------------
# Presets  (sd_in / sd_out = 0.1 by default; off-diagonal input means = 0)
# --------------------------------------------------------------------------
def default_codes(baseline=None):
    a = affine(VALUES, baseline)
    one = np.array([1.0])
    rec_in = a[0]                                  # recurrent preset: equal input = a(0%) ...
    mu_rec = 1 - rec_in / a                        # ... and gain chosen so r = in/(1-g) = a(v)
    return [
        NetCode("Readout learning", "value in readout weights",
                sizes=np.ones(3), mu_in=np.eye(3), mu_out=a),
        NetCode("Subpopulation recruitment", "value in number of active units",
                sizes=a, mu_in=np.eye(3), mu_out=np.ones(3)),
        NetCode("Level-based coding\n(input-driven, subgroups)", "group level set by input weights",
                sizes=np.ones(3), mu_in=np.diag(a), mu_out=np.ones(3)),
        NetCode("Level-based coding\n(recurrently-driven, subgroups)",
                "group level set by recurrent amplification",
                sizes=np.ones(3), mu_in=rec_in * np.eye(3), mu_out=np.ones(3), mu_rec=mu_rec),
        NetCode("Level-based coding\n(input-driven, single population)",
                "mixed selectivity; input strength\ngraded by value",
                sizes=one, mu_in=a[None, :], sd_in=0.3, mu_out=one),
        NetCode("Distributed features +\nreadout learning",
                "mixed selectivity, equal input;\nlearned readout",
                sizes=one, mu_in=np.full((1, 3), 0.5), sd_in=0.3, mu_out=one, readout="fit"),
    ]


CODES = default_codes()


# --------------------------------------------------------------------------
# Drawing: network schematic
# --------------------------------------------------------------------------
X_IN, X_HID, X_OUT = 0.0, 1.7, 3.4
Y_ROWS = np.array([1.2, 0.0, -1.2])
R_IN, R_OUT = 0.17, 0.24


def _lw(w, wmax):
    return MIN_LW + (MAX_LW - MIN_LW) * min(abs(w) / wmax, 1)


def _edge(ax, p0, p1, r0, r1, lw, color, z=1, alpha=1.0):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = (p1 - p0) / np.linalg.norm(p1 - p0)
    ax.add_patch(FancyArrowPatch(p0 + d * r0, p1 - d * r1, arrowstyle="-|>",
                                 mutation_scale=7 + 0.8 * lw, lw=lw, color=color, alpha=alpha,
                                 shrinkA=0, shrinkB=0, zorder=z, joinstyle="miter", capstyle="butt"))


def _self_loop(ax, c, r, w, wmax):
    if w <= 0.01:
        return
    lw = 1.0 + 2.5 * min(abs(w) / wmax, 1)
    a0, a1 = np.deg2rad(165), np.deg2rad(110)
    p0 = c + r * np.array([np.cos(a0), np.sin(a0)])
    p1 = c + r * np.array([np.cos(a1), np.sin(a1)])
    ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle="arc3,rad=-1.7", arrowstyle="-|>",
                                 mutation_scale=8 + 2 * lw, lw=lw, color="#555555", zorder=4))


def _shade(level):
    return LEVEL_CMAP(Normalize(0, LEVEL_VMAX, clip=True)(level))


def _edge_weights(code, sim):
    """Mean input weights (G,3), mean readout per group (G,) or, for a single population
    with a learned readout, the alignment of w_out with each stimulus pattern (3,)."""
    mu_in, mu_out, mu_rec = code._arr()
    if code.readout == "fit":
        w, g = sim["net"]["w_out"], sim["net"]["group"]
        if code.G == 1:
            R = sim["R"]
            cos = R @ w / (np.linalg.norm(R, axis=1) * np.linalg.norm(w) + 1e-12)
            out = np.clip(cos, 0, None)
            out = out / max(out.max(), 1e-9)
        else:
            out = np.array([w[g == k].mean() for k in range(code.G)])
            out = out / max(np.abs(out).max(), 1e-9)
    else:
        out = mu_out
    return mu_in, out, mu_rec


def draw_network(ax, code, sim=None, wmax=None):
    sim = code.simulate() if sim is None else sim
    mu_in, out, mu_rec = _edge_weights(code, sim)
    if wmax is None:
        wmax = max(np.abs(mu_in).max(), np.abs(out).max(), 1e-9)
    for i, y in enumerate(Y_ROWS):
        ax.add_patch(Circle((X_IN, y), R_IN, fc=STIM_COLORS[i], ec="k", lw=1, zorder=3))
        ax.text(X_IN - 0.25, y, STIM_NAMES[i], ha="right", va="center", fontsize=10,
                color=STIM_COLORS[i], fontweight="bold")
    ax.add_patch(Circle((X_OUT, 0), R_OUT, fc="#f2e8cf", ec="k", lw=1.2, zorder=3))
    ax.text(X_OUT, 0, r"$\hat{V}$", ha="center", va="center", fontsize=13, zorder=4)

    if code.G == 3:
        size = np.clip(np.atleast_1d(np.asarray(code.sizes, float)), 0, None)
        radii = GROUP_MAX_RADIUS * np.sqrt(size / max(size.max(), 1e-9))  # area ~ group size parameter
        centres = [np.array([X_HID, y]) for y in Y_ROWS]
        for g in range(3):
            c, r = centres[g], max(radii[g], 0.02)
            ax.add_patch(Circle(c, r, fc=_shade(sim["level"][g]), ec=STIM_COLORS[g], lw=2.4, zorder=3))
            for s in range(3):                      # input edges stimulus s -> group g
                if abs(mu_in[g, s]) >= EDGE_MIN:
                    _edge(ax, (X_IN, Y_ROWS[s]), c, R_IN, r, _lw(mu_in[g, s], wmax), STIM_COLORS[s],
                          alpha=1.0 if s == g else 0.7)
            _edge(ax, c, (X_OUT, 0), r, R_OUT, _lw(out[g], wmax), "#333333")
            _self_loop(ax, c, r, mu_rec[g], 1.0)
    else:
        R = MAX_RADIUS
        c = np.array([X_HID, 0.0])
        ax.add_patch(Circle(c, R, fc="none", ec="k", lw=2.0, zorder=4))
        band_h = 2 * R / 3
        tops = c[1] + R - band_h * np.arange(3)
        for i in range(3):
            rect = Rectangle((c[0] - R, tops[i] - band_h), 2 * R, band_h,
                             fc=_shade(sim["level"][i]), ec="none", zorder=2)
            ax.add_patch(rect)
            rect.set_clip_path(Circle(c, R, transform=ax.transData))
            yb = tops[i] - band_h / 2
            if i:
                ax.plot([c[0] - R, c[0] + R], [tops[i], tops[i]], ls=":", color="#888", lw=0.8, zorder=3)
            xl = c[0] - np.sqrt(R**2 - (yb - c[1])**2)
            if abs(mu_in[0, i]) >= EDGE_MIN:
                _edge(ax, (X_IN, Y_ROWS[i]), (xl, yb), R_IN, 0, _lw(mu_in[0, i], wmax), STIM_COLORS[i])
        if code.readout == "fit":                    # alignment of learned readout per state
            for i in range(3):
                yb = tops[i] - band_h / 2
                xr = c[0] + np.sqrt(R**2 - (yb - c[1])**2)
                _edge(ax, (xr, yb), (X_OUT, 0), 0, R_OUT, _lw(out[i] * wmax, wmax), STIM_COLORS[i])
        else:
            _edge(ax, c, (X_OUT, 0), R, R_OUT, _lw(out[0], wmax), "#333333")
            _self_loop(ax, c, R, mu_rec[0], 1.0)
        ax.text(c[0], -1.42, "single population; bands = state per stimulus",
                ha="center", va="top", fontsize=7, color="#555")

    ax.set_title(code.name, fontsize=11, pad=4)
    if code.note:
        ax.text((X_IN + X_OUT) / 2, 1.85, code.note, ha="center", va="bottom",
                fontsize=8, style="italic", color="#444")
    ax.set_xlim(X_IN - 0.75, X_OUT + 0.45)
    ax.set_ylim(-1.75, 2.25)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_table(ax, features):
    x0, x1, dy = 0.0, 1.0, 1.0
    for j, (col, val) in enumerate(zip(FEATURE_COLS, features)):
        y = -j * dy
        if j % 2 == 0:
            ax.add_patch(Rectangle((x0, y - dy / 2), x1 - x0, dy, fc="#f3f3f3", ec="none", zorder=0))
        ax.text(x0 + 0.02, y, col, ha="left", va="center", fontsize=8, color="#333")
        colr = {"Y": "#2a7a2a", "N": "#b00020"}.get(val, "#777")
        ax.text(x1 - 0.03, y, val, ha="right", va="center", fontsize=10, fontweight="bold", color=colr)
    ax.plot([x0, x1], [dy / 2, dy / 2], color="#bbb", lw=0.8)
    ax.set_xlim(x0, x1)
    ax.set_ylim(-3 * dy - dy / 2, dy / 2 + 0.05)
    ax.axis("off")


def _sort_units(md):
    """Same ordering as transfer heatmap_seed: by preferred stimulus, then peak; silent last."""
    peak = md.max(0); pref = md.argmax(0); silent = peak < 0.05 * max(peak.max(), 1e-12)
    return np.lexsort((-peak, np.where(silent, 3, pref)))


def draw_heatmap(ax, R, vmax=None, ylabel=True):
    vmax = HEAT_VMAX if vmax is None else vmax
    im = ax.imshow(R[:, _sort_units(R)].T, aspect="auto", cmap=HEAT_CMAP, vmin=0, vmax=vmax,
                   interpolation="nearest")
    ax.set_xticks(range(3))
    ax.set_xticklabels(STIM_NAMES, fontsize=9)
    for t, c in zip(ax.get_xticklabels(), STIM_COLORS):
        t.set_color(c); t.set_fontweight("bold")
    if ylabel:
        ax.set_ylabel("hidden unit (sorted)", fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
    else:
        ax.set_yticks([])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return im


def draw_vhat(ax, vhat):
    fitted, r2, slope = readout_fit(vhat)
    ok = r2 > 0.99 and slope > 0
    ax.bar(range(3), vhat, color=STIM_COLORS, edgecolor="k", lw=0.8)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(3)); ax.set_xticklabels(STIM_NAMES)
    ax.set_ylabel(r"$\hat V = w_{out}\cdot r\,/\,N$", fontsize=9)
    ax.set_title(f"readout  (affine fit R² = {r2:.3f})", fontsize=9,
                 color="#2a7a2a" if ok else "#b00020")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def draw_responders(ax, R):
    n = responder_counts(R)
    cols = STIM_COLORS + ["#bbbbbb"] * 4 + ["#888888"]
    ax.bar(range(8), n, color=cols, edgecolor="k", lw=0.6)
    ax.set_xticks(range(8)); ax.set_xticklabels(RESP_LABELS + ["none"], fontsize=7)
    ax.set_ylabel("# units", fontsize=9)
    ax.set_title(f"responder categories (r > {RESP_THRESH:g})", fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def make_figure(codes=None, fname=None, ncols=None, show_table=None, show_heatmap=None):
    """Grid of codes (schematic / heatmap / feature table). Saves png/pdf/svg to OUT_DIR
    if fname is given. Returns the Figure."""
    codes = default_codes() if codes is None else codes
    show_table = SHOW_FEATURES if show_table is None else show_table
    show_heatmap = SHOW_HEATMAPS if show_heatmap is None else show_heatmap
    ncols = ncols or len(codes)
    nblocks = int(np.ceil(len(codes) / ncols))
    sims = [c.simulate() for c in codes]
    wmax = max(max(np.abs(_edge_weights(c, s)[0]).max(), np.abs(_edge_weights(c, s)[1]).max())
               for c, s in zip(codes, sims))

    rows = [("net", 3.7)] + ([("heat", 2.4)] if show_heatmap else []) + \
           ([("table", 1.3)] if show_table else [])
    ratios = [h for _, h in rows] * nblocks + [0.55]
    fig = plt.figure(figsize=(3.4 * ncols, sum(ratios) * 0.95))
    gs = fig.add_gridspec(len(ratios), ncols, height_ratios=ratios, hspace=0.22, wspace=0.12,
                          left=0.04, right=0.98, top=0.96, bottom=0.02)
    im = None
    for k, (code, sim) in enumerate(zip(codes, sims)):
        b, col = divmod(k, ncols)
        for r, (kind, _) in enumerate(rows):
            cell = gs[b * len(rows) + r, col]
            if kind == "net":
                draw_network(fig.add_subplot(cell), code, sim, wmax)
            elif kind == "heat":
                sub = cell.subgridspec(1, 3, width_ratios=[1.0, 1.6, 0.7])
                im = draw_heatmap(fig.add_subplot(sub[0, 1]), sim["R"], ylabel=(col == 0))
            else:
                sub = cell.subgridspec(1, 3, width_ratios=[0.03, 1, 0.03])
                draw_table(fig.add_subplot(sub[0, 1]), infer_features(code, sim))
        _, r2, slope = readout_fit(sim["vhat"])
        if not (r2 > 0.99 and slope > 0):
            print(f"NOTE: '{code.name}' readout not affine in value (R^2 = {r2:.3f})")

    leg = gs[-1, :].subgridspec(2, 7, width_ratios=[0.2, 1, 0.25, 1, 0.25, 3, 0.2],
                                height_ratios=[1, 1.4])
    cb = fig.colorbar(ScalarMappable(Normalize(0, LEVEL_VMAX), LEVEL_CMAP),
                      cax=fig.add_subplot(leg[0, 1]), orientation="horizontal")
    cb.set_label("schematic: mean group activity", fontsize=8); cb.ax.tick_params(labelsize=7)
    if show_heatmap and im is not None:
        cb2 = fig.colorbar(im, cax=fig.add_subplot(leg[0, 3]), orientation="horizontal")
        cb2.set_label("heatmap: unit activity", fontsize=8); cb2.ax.tick_params(labelsize=7)
    tax = fig.add_subplot(leg[:, 5]); tax.axis("off")
    tax.text(0, 0.75, "node area ∝ group size   ·   edge width ∝ |mean weight|   ·   "
             "loop = within-group recurrent gain", fontsize=8.5, va="center", color="#333")
    if fname:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf", "svg"):
            fig.savefig(OUT_DIR / f"{fname}.{ext}", dpi=200, bbox_inches="tight")
    return fig


def explore(code, wmax=None):
    """One code in detail: schematic | heatmap | V_hat | responder categories | features."""
    sim = code.simulate()
    fig = plt.figure(figsize=(17, 3.9))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.35, 0.7, 0.8, 1.1, 1.0], wspace=0.45,
                          left=0.02, right=0.99, top=0.86, bottom=0.17)
    draw_network(fig.add_subplot(gs[0]), code, sim, wmax)
    ax = fig.add_subplot(gs[1])
    im = draw_heatmap(ax, sim["R"])
    fig.colorbar(im, ax=ax, fraction=0.08, pad=0.04).ax.tick_params(labelsize=7)
    draw_vhat(fig.add_subplot(gs[2]), sim["vhat"])
    draw_responders(fig.add_subplot(gs[3]), sim["R"])
    tax = fig.add_subplot(gs[4].subgridspec(3, 1, height_ratios=[0.6, 1, 0.6])[1])
    draw_table(tax, infer_features(code, sim))
    tax.set_title("features (inferred)", fontsize=9)
    return fig


if __name__ == "__main__":
    plt.close(make_figure(CODES, "value_code_solutions"))
    print(OUT_DIR / "value_code_solutions.png")
    plt.close(make_figure(CODES, "value_code_solutions_plus_distributed", ncols=3))
    print(OUT_DIR / "value_code_solutions_plus_distributed.png")
