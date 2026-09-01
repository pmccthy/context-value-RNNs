from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class RNN(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        recurrent_gain=0.9,
        h2h_init="orthogonal",
        init_scale=1.0,
        input_plastic=True,
        hidden_plastic=True,
        output_plastic=True,
        dynamics="elman",
    ):
        """Initialise a vanilla Elman RNN.

        Args:
            input_size: Dimensionality of the input at each time step.
            hidden_size: Number of recurrent units.
            output_size: Dimensionality of the readout.
            recurrent_gain: Spectral radius of the initial recurrent weight matrix.
                Used only when h2h_init="orthogonal".
            h2h_init: Initialisation scheme for the recurrent weight matrix.
                "orthogonal" — orthogonal matrix scaled by recurrent_gain (default).
                "kaiming"    — Kaiming normal, same as the input/output layers.
            init_scale: Global multiplier applied to all weights after initialisation.
                Values < 1 give smaller initial activations; the network must grow
                its weights during training rather than suppress them.
            input_plastic: Whether input weights are trainable.
            hidden_plastic: Whether recurrent weights are trainable.
            output_plastic: Whether readout weights are trainable.
            dynamics: "elman" (default, unchanged) -- h_t = phi(W_in x_t +
                W_rec h_{t-1}), nonlinearity applied to the SUM of input and
                recurrent drive; the persisted/returned state IS the rate
                (already post-nonlinearity).
                "mastrogiuseppe" -- Mastrogiuseppe & Ostojic (2018) ordering:
                the nonlinearity is applied to each unit's PREVIOUS state
                before the recurrent weights multiply/sum it (rate_prev =
                phi(h_prev); recurrent = W_rec @ rate_prev), and the input
                is added SEPARATELY, never itself passed through phi:
                h_t = W_in x_t + W_rec @ phi(h_prev). The persisted/returned
                state h is therefore the raw, unbounded PRE-nonlinearity
                "current" (can be negative) -- use self.activity(h) to get
                the rate phi(h) for readout/recorded-activation purposes
                (see self.activity and RNN.forward, which already does
                this correctly).
        """
        super().__init__()

        if dynamics not in ("elman", "mastrogiuseppe"):
            raise ValueError(f"dynamics must be 'elman' or 'mastrogiuseppe', got {dynamics!r}")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.recurrent_gain = recurrent_gain
        self.h2h_init = h2h_init
        self.init_scale = init_scale
        self.dynamics = dynamics
        self.nonlinearity = nn.ReLU()

        self.input2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

        self.input2h.requires_grad_(input_plastic)
        self.h2h.requires_grad_(hidden_plastic)
        self.h2o.requires_grad_(output_plastic)

        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming init for input/output layers; h2h init determined by self.h2h_init."""
        for layer in [self.input2h, self.h2o]:
            nn.init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        if self.h2h_init == "kaiming":
            nn.init.kaiming_normal_(self.h2h.weight)
        else:
            nn.init.orthogonal_(self.h2h.weight, gain=self.recurrent_gain)
        nn.init.zeros_(self.h2h.bias)
        if self.init_scale != 1.0:
            with torch.no_grad():
                for layer in [self.input2h, self.h2h, self.h2o]:
                    layer.weight.mul_(self.init_scale)

    def init_hidden(self, batch_size, device):
        """Return a zero initial hidden state.

        Args:
            batch_size: Number of sequences in the batch.
            device: Target device for the tensor.

        Returns:
            Tensor of shape (batch_size, hidden_size) filled with zeros.
        """
        return torch.zeros(batch_size, self.hidden_size, device=device)

    def activity(self, h):
        """Map the persisted recurrent state to the "activity"/rate used for
        readout and for recorded/extracted activations. Identity for the
        default "elman" dynamics (h is already post-nonlinearity); phi(h)
        for "mastrogiuseppe" dynamics (h is the raw pre-nonlinearity
        "current", and the rate r=phi(h) is the M&O-convention firing
        rate -- see the dynamics= docstring in __init__). ALWAYS call this
        (not the raw state) wherever a linear readout or a recorded
        "activation" is needed, so both dynamics conventions stay
        interchangeable everywhere downstream."""
        if self.dynamics == "mastrogiuseppe":
            return self.nonlinearity(h)
        return h

    def recurrence(self, x_t, h_prev):
        """Compute one step of the recurrent dynamics.

        Args:
            x_t: Input at the current time step, shape (batch, input_size).
            h_prev: Hidden state from the previous time step, shape (batch, hidden_size).

        Returns:
            h_t: Updated hidden state, shape (batch, hidden_size).
        """
        if self.dynamics == "mastrogiuseppe":
            rate_prev = self.nonlinearity(h_prev)
            h_t = self.input2h(x_t) + self.h2h(rate_prev)
        else:
            h_t = self.nonlinearity(self.input2h(x_t) + self.h2h(h_prev))
        return h_t

    def forward(self, x, hidden=None):
        """Run the RNN over a full input sequence.

        Args:
            x: Input tensor of shape (batch, time_steps, input_size).
            hidden: Optional initial hidden state of shape (batch, hidden_size).
                Defaults to zeros when None.

        Returns:
            output: Readout tensor of shape (batch, time_steps, output_size).
            hidden_all: ACTIVITY (self.activity(h), not the raw state)
                tensor of shape (batch, time_steps, hidden_size) -- for
                "elman" dynamics this is identical to the raw state as
                before; for "mastrogiuseppe" dynamics this is phi(h), the
                rate.
        """
        if hidden is None:
            hidden = self.init_hidden(x.shape[0], x.device)

        hidden_all = []
        for t in range(x.size(1)):
            hidden = self.recurrence(x[:, t, :], hidden)
            hidden_all.append(self.activity(hidden))

        hidden_all = torch.stack(hidden_all, dim=1)  # (batch, time, hidden)
        output = self.h2o(hidden_all)

        return output, hidden_all

    def get_activations(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[np.ndarray, float]:
        self.eval()
        with torch.no_grad():
            outputs, hidden_states = self(X.transpose(0, 1))
            outputs_flat = outputs[mask].reshape(-1, outputs.shape[-1])
            targets_flat = y.transpose(0, 1)[mask].reshape(-1, y.shape[-1])
            accuracy = (outputs_flat.argmax(1) == targets_flat.argmax(1)).float().mean().item()
            activations = hidden_states.transpose(0, 1).cpu().numpy()
        return activations, accuracy

class LeakyRNN(RNN):

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        tau,
        dt=1.0,
        recurrent_gain=0.9,
        init_scale=1.0,
        input_plastic=True,
        hidden_plastic=True,
        output_plastic=True,
        dynamics="elman",
    ):
        """Initialise a leaky (continuous-time) Elman RNN.

        The hidden state update follows:
            h_t = (1 - alpha) * h_{t-1} + alpha * phi(W_in x_t + W_rec h_{t-1})
        where alpha = dt / tau (dynamics="elman"; see RNN.__init__'s
        dynamics= docstring for the "mastrogiuseppe" alternative, which
        leaks the raw pre-nonlinearity current rather than the rate -- THIS
        is the only setting under which "mastrogiuseppe" actually differs
        numerically from "elman": with alpha=1 (tau=dt, no leak) the two
        dynamics conventions are mathematically IDENTICAL regardless of
        nonlinearity placement (verified analytically and numerically --
        see cxval.models module docstring / git history around 17_07_26).

        Args:
            input_size: Dimensionality of the input at each time step.
            hidden_size: Number of recurrent units.
            output_size: Dimensionality of the readout.
            tau: Time constant of the leak; larger values give slower dynamics.
                tau == dt recovers the memoryless (no-leak) map.
            dt: Simulation timestep. Defaults to 1.0.
            recurrent_gain: Spectral radius for orthogonal h2h initialisation.
            init_scale: Global multiplier applied to all weights after
                initialisation (same convention as RNN.__init__; previously
                missing from LeakyRNN, added so build_backbone's tau= path
                gets the same init scaling as the non-leaky path).
            input_plastic: Whether input weights are trainable.
            hidden_plastic: Whether recurrent weights are trainable.
            output_plastic: Whether readout weights are trainable.
            dynamics: "elman" (default) or "mastrogiuseppe" -- see RNN.__init__.
        """
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            recurrent_gain=recurrent_gain,
            init_scale=init_scale,
            input_plastic=input_plastic,
            hidden_plastic=hidden_plastic,
            output_plastic=output_plastic,
            dynamics=dynamics,
        )

        self.tau = tau
        self.dt = dt
        self.alpha = dt / tau

    def recurrence(self, x_t, h_prev):
        """Compute one leaky-integration step.

        Args:
            x_t: Input at the current time step, shape (batch, input_size).
            h_prev: Hidden state from the previous time step, shape (batch, hidden_size).

        Returns:
            h_t: Updated hidden state, shape (batch, hidden_size).
        """
        if self.dynamics == "mastrogiuseppe":
            # Leak the RAW current (matches the literal M&O ODE tau*dx/dt =
            # -x + W*phi(x) + I); h_new here is the un-nonlinearised new
            # current, not a rate.
            rate_prev = self.nonlinearity(h_prev)
            h_new = self.input2h(x_t) + self.h2h(rate_prev)
        else:
            h_new = self.nonlinearity(self.input2h(x_t) + self.h2h(h_prev))
        h_t = (1 - self.alpha) * h_prev + self.alpha * h_new
        return h_t


class LowRankRNN(RNN):
    """RNN whose recurrent weight matrix is constrained to rank R.

    The recurrent connectivity is:
        W_rec = J_0 + M @ N^T / hidden_size
    where M and N are each of shape (hidden_size, rank), following
    Mastrogiuseppe & Ostojic (2018, Neuron).

    J_0 is a fixed random matrix drawn from N(0, gain² / hidden_size) at
    initialisation and stored as a non-trainable buffer.  Setting gain=0
    (default) removes the random component and recovers the pure low-rank model.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        rank,
        gain=0.0,
        input_plastic=True,
        hidden_plastic=True,
        output_plastic=True,
        dynamics="elman",
    ):
        """Initialise a low-rank RNN.

        Args:
            input_size: Dimensionality of the input at each time step.
            hidden_size: Number of recurrent units.
            output_size: Dimensionality of the readout.
            rank: Rank R of the recurrent weight matrix.
            gain: Scaling factor for the fixed random component J_0; each
                entry is drawn from N(0, gain² / hidden_size).  gain=0
                disables the random component entirely.
            input_plastic: Whether input weights are trainable.
            hidden_plastic: Whether low-rank factors (m, n) are trainable.
            output_plastic: Whether readout weights are trainable.
            dynamics: "elman" (default) or "mastrogiuseppe" -- see RNN.__init__.
        """
        # hidden_plastic=True here only so the parent creates h2h without
        # error; we remove it immediately afterwards.
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            input_plastic=input_plastic,
            hidden_plastic=True,
            output_plastic=output_plastic,
            dynamics=dynamics,
        )

        del self.h2h

        self.rank = rank
        self.gain = gain

        # Low-rank factors initialised from N(0, 1); the 1/N scaling is
        # applied in the recurrence, matching Mastrogiuseppe & Ostojic (2018).
        self.m = nn.Parameter(
            torch.randn(hidden_size, rank), requires_grad=hidden_plastic
        )
        self.n = nn.Parameter(
            torch.randn(hidden_size, rank), requires_grad=hidden_plastic
        )

        if gain > 0:
            J0 = torch.randn(hidden_size, hidden_size) * gain / (hidden_size**0.5)
            self.register_buffer("J0", J0)
        else:
            self.J0 = None

    def _initialize_weights(self):
        """Initialise input and readout layers; low-rank factors are set in __init__."""
        for layer in [self.input2h, self.h2o]:
            nn.init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def recurrence(self, x_t, h_prev):
        """Compute one step with low-rank recurrent dynamics.

        Args:
            x_t: Input at the current time step, shape (batch, input_size).
            h_prev: Hidden state from the previous time step, shape (batch, hidden_size).

        Returns:
            h_t: Updated hidden state, shape (batch, hidden_size).
        """
        if self.dynamics == "mastrogiuseppe":
            rate_prev = self.nonlinearity(h_prev)
            lr_drive = (rate_prev @ self.n) @ self.m.T / self.hidden_size
            recurrent = lr_drive if self.J0 is None else lr_drive + rate_prev @ self.J0.T
            h_t = self.input2h(x_t) + recurrent
        else:
            lr_drive = (h_prev @ self.n) @ self.m.T / self.hidden_size
            recurrent = lr_drive if self.J0 is None else lr_drive + h_prev @ self.J0.T
            h_t = self.nonlinearity(self.input2h(x_t) + recurrent)
        return h_t


class LowRankLeakyRNN(LowRankRNN):
    """Low-rank RNN with leaky (continuous-time) integration.

    Combines the rank-R recurrent connectivity of :class:`LowRankRNN` with
    the leaky hidden-state update of :class:`LeakyRNN`:
        h_t = (1 - alpha) * h_{t-1} + alpha * phi(W_in x_t + J h_{t-1})
    where J = J_0 + M @ N^T / hidden_size.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        rank,
        tau,
        dt=1.0,
        gain=0.0,
        input_plastic=True,
        hidden_plastic=True,
        output_plastic=True,
        dynamics="elman",
    ):
        """Initialise a low-rank leaky RNN.

        Args:
            input_size: Dimensionality of the input at each time step.
            hidden_size: Number of recurrent units.
            output_size: Dimensionality of the readout.
            rank: Rank R of the recurrent weight matrix.
            tau: Time constant of the leak; larger values give slower dynamics.
            dt: Simulation timestep. Defaults to 1.0.
            gain: Scaling factor for the fixed random component J_0; each
                entry is drawn from N(0, gain² / hidden_size).  gain=0
                disables the random component entirely.
            input_plastic: Whether input weights are trainable.
            hidden_plastic: Whether low-rank factors (m, n) are trainable.
            output_plastic: Whether readout weights are trainable.
            dynamics: "elman" (default) or "mastrogiuseppe" -- see RNN.__init__.
        """
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            rank=rank,
            gain=gain,
            input_plastic=input_plastic,
            hidden_plastic=hidden_plastic,
            output_plastic=output_plastic,
            dynamics=dynamics,
        )

        self.tau = tau
        self.dt = dt
        self.alpha = dt / tau

    def recurrence(self, x_t, h_prev):
        """Compute one leaky step with low-rank recurrent dynamics.

        Args:
            x_t: Input at the current time step, shape (batch, input_size).
            h_prev: Hidden state from the previous time step, shape (batch, hidden_size).

        Returns:
            h_t: Updated hidden state, shape (batch, hidden_size).
        """
        if self.dynamics == "mastrogiuseppe":
            rate_prev = self.nonlinearity(h_prev)
            lr_drive = (rate_prev @ self.n) @ self.m.T / self.hidden_size
            recurrent = lr_drive if self.J0 is None else lr_drive + rate_prev @ self.J0.T
            h_new = self.input2h(x_t) + recurrent
        else:
            lr_drive = (h_prev @ self.n) @ self.m.T / self.hidden_size
            recurrent = lr_drive if self.J0 is None else lr_drive + h_prev @ self.J0.T
            h_new = self.nonlinearity(self.input2h(x_t) + recurrent)
        h_t = (1 - self.alpha) * h_prev + self.alpha * h_new
        return h_t


def build_backbone(
    rank,
    input_size,
    hidden_size,
    output_size=1,
    recurrent_gain=0.9,
    init_scale=0.02,
    lowrank_gain=0.0,
    lowrank_scale=1.0,
    input_plastic=True,
    hidden_plastic=True,
    output_plastic=True,
    dynamics="elman",
    tau=None,
    dt=1.0,
):
    """Construct a backbone RNN for a given nominal rank, for a rank-sweep comparison.

    Deliberately uses the EXISTING :class:`RNN` / :class:`LowRankRNN` classes
    unchanged (no new recurrent parameterisation) so low-rank connectivity
    keeps the standard, literature-standard M/N factorisation (Mastrogiuseppe
    & Ostojic 2018) — comparable across models via gauge-invariant post-hoc
    analysis (see ``cxval.analysis``) rather than via architecture.

    Args:
        rank: 1, 2, 3, ... for a :class:`LowRankRNN`; or ``None`` / ``"full"``
            / any value >= hidden_size for a full-rank (dense) :class:`RNN`.
        input_size, hidden_size, output_size: Layer sizes (output_size is
            unused by the vigour heads but required by the RNN constructors).
        recurrent_gain, init_scale: Passed to the full-rank RNN's orthogonal
            h2h init (matches the FENS-2026 final config: recurrent_gain=0.9,
            init_scale=0.02).
        lowrank_gain: Passed to LowRankRNN's fixed random full-rank component
            J0 (default 0 = pure low-rank, no J0).
        lowrank_scale: Multiplies LowRankRNN's (m, n) after construction.
            NOTE (init asymmetry): LowRankRNN draws m, n ~ N(0,1) with NO
            scaling by init_scale/recurrent_gain -- that's how the existing
            class already works. This means, out of the box, a low-rank
            model's initial recurrent drive magnitude will generally differ
            from the full-rank model's (which IS scaled by init_scale and
            recurrent_gain). If you want comparable initial recurrent-drive
            scale across the rank sweep, tune lowrank_scale (e.g. via a
            quick numerical check of ||h_prev @ W_rec|| at init) rather than
            assuming it matches by default.
        input_plastic, hidden_plastic, output_plastic: Passed through.
        dynamics: "elman" (default, unchanged behaviour) or "mastrogiuseppe"
            -- see RNN.__init__'s dynamics= docstring. Applied identically
            to both the full-rank and low-rank branch, so a rank sweep
            never mixes two different recurrence conventions. IMPORTANT:
            with tau=None (default, no leak), "mastrogiuseppe" is
            mathematically IDENTICAL to "elman" -- there is no discrete,
            memoryless ordering of (nonlinearity, weights, input) that
            differs from a standard Elman map (proven by induction: both
            reduce to rate_t = phi(W_in x_t + W_rec rate_{t-1}) from a zero
            initial state). To get a genuinely different (M&O-style leaky)
            network, pass tau > dt.
        tau: if given (and > dt), builds the LEAKY variant (LeakyRNN /
            LowRankLeakyRNN) instead of the memoryless one -- REQUIRED for
            dynamics="mastrogiuseppe" to have any effect. None (default) =
            memoryless map (dynamics has no effect either way).
        dt: simulation timestep for the leaky variant (only used if tau is
            given). Default 1.0.

    Returns:
        An RNN/LeakyRNN or LowRankRNN/LowRankLeakyRNN instance (all are
        ``cxval.models.RNN`` subclasses with an identical
        recurrence/forward/step interface).
    """
    is_full = (
        rank is None
        or (isinstance(rank, str) and rank.lower() == "full")
        or int(rank) >= hidden_size
    )
    leaky = tau is not None and tau != dt
    if is_full:
        if leaky:
            return LeakyRNN(
                input_size=input_size, hidden_size=hidden_size, output_size=output_size,
                tau=tau, dt=dt, recurrent_gain=recurrent_gain, init_scale=init_scale,
                input_plastic=input_plastic, hidden_plastic=hidden_plastic,
                output_plastic=output_plastic, dynamics=dynamics,
            )
        return RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            recurrent_gain=recurrent_gain,
            h2h_init="orthogonal",
            init_scale=init_scale,
            input_plastic=input_plastic,
            hidden_plastic=hidden_plastic,
            output_plastic=output_plastic,
            dynamics=dynamics,
        )
    if leaky:
        backbone = LowRankLeakyRNN(
            input_size=input_size, hidden_size=hidden_size, output_size=output_size,
            rank=int(rank), tau=tau, dt=dt, gain=lowrank_gain,
            input_plastic=input_plastic, hidden_plastic=hidden_plastic,
            output_plastic=output_plastic, dynamics=dynamics,
        )
    else:
        backbone = LowRankRNN(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            rank=int(rank),
            gain=lowrank_gain,
            input_plastic=input_plastic,
            hidden_plastic=hidden_plastic,
            output_plastic=output_plastic,
            dynamics=dynamics,
        )
    if lowrank_scale != 1.0:
        with torch.no_grad():
            backbone.m.mul_(lowrank_scale)
            backbone.n.mul_(lowrank_scale)
    return backbone


class ActorCritic(nn.Module):
    """Actor-critic wrapper around any backbone RNN.

    Takes a pre-constructed backbone (any subclass of :class:`RNN`) and adds
    separate linear policy and value heads that read from the hidden states.
    The backbone's own ``h2o`` readout layer is unused.

    Args:
        backbone: A constructed RNN instance (e.g. LeakyRNN, LowRankRNN).
        num_actions: Number of discrete actions for the policy head.
        policy_clip: When > 0, action probabilities are clamped to
            [policy_clip, 1 - policy_clip] before sampling.
        readout_fraction: Fraction of hidden units that project to the policy
            and value heads (default 1.0 = full readout).  Only the first
            ``int(hidden_size * readout_fraction)`` neurons are connected to
            the output heads; the remaining units participate in recurrent
            dynamics but have no direct output projection.  Set to 0.5 to
            match the cogNN partial-readout architecture.
    """

    def __init__(
        self,
        backbone: RNN,
        num_actions: int,
        policy_clip: float = 0.0,
        readout_fraction: float = 1.0,
    ):
        super().__init__()
        self.backbone = backbone
        self.readout_fraction = readout_fraction
        self.n_readout = max(1, int(backbone.hidden_size * readout_fraction))
        self.policy_head = nn.Linear(self.n_readout, num_actions)
        self.value_head = nn.Linear(self.n_readout, 1)
        self.policy_clip = policy_clip

    def make_dist(self, logits):
        """Build a Categorical distribution from logits, optionally clamping probabilities.

        When policy_clip > 0, action probabilities are clamped to
        [policy_clip, 1 - policy_clip] and renormalised before sampling.
        This prevents any action from reaching probability 0 and ensures
        ongoing exploration even after many gradient updates.
        """
        if self.policy_clip > 0.0:
            probs = torch.softmax(logits, dim=-1)
            probs = probs.clamp(min=self.policy_clip)
            probs = probs / probs.sum(dim=-1, keepdim=True)
            return Categorical(probs=probs)
        return Categorical(logits=logits)

    def forward(self, x, hidden=None):
        """Run the backbone and compute policy logits and value estimates.

        Args:
            x: Input tensor of shape (batch, time_steps, input_size).
            hidden: Optional initial hidden state of shape (batch, hidden_size).

        Returns:
            logits: Policy logits of shape (batch, time_steps, num_actions).
            values: Value estimates of shape (batch, time_steps).
            hidden_all: Hidden states of shape (batch, time_steps, hidden_size).
        """
        _, hidden_all = self.backbone(x, hidden)
        readout = hidden_all[..., :self.n_readout]
        logits = self.policy_head(readout)
        values = self.value_head(readout).squeeze(-1)
        return logits, values, hidden_all

    def step(self, obs, hidden=None):
        """Single-timestep forward pass for online environment interaction.

        Args:
            obs: Observation tensor of shape (batch, input_size).
            hidden: Current hidden state of shape (batch, hidden_size), or None.

        Returns:
            logits: Policy logits of shape (batch, num_actions).
            value: Value estimate of shape (batch,).
            hidden: Updated hidden state of shape (batch, hidden_size).
        """
        if hidden is None:
            hidden = self.backbone.init_hidden(obs.shape[0], obs.device)
        hidden = self.backbone.recurrence(obs, hidden)
        readout = self.backbone.activity(hidden)[..., :self.n_readout]
        logits = self.policy_head(readout)
        value = self.value_head(readout).squeeze(-1)
        return logits, value, hidden

    def get_activations(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[np.ndarray, float]:
        self.eval()
        with torch.no_grad():
            logits, _, hidden_states = self(X.transpose(0, 1))
            logits_flat = logits[mask].reshape(-1, logits.shape[-1])
            targets_flat = y.transpose(0, 1)[mask].reshape(-1, y.shape[-1])
            accuracy = (logits_flat.argmax(1) == targets_flat.argmax(1)).float().mean().item()
            activations = hidden_states.transpose(0, 1).cpu().numpy()
        return activations, accuracy
