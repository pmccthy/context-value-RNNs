import torch
from torch import nn


class RNN(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        input_plastic=True,
        hidden_plastic=True,
        output_plastic=True,
    ):
        """Initialise a vanilla Elman RNN.

        Args:
            input_size: Dimensionality of the input at each time step.
            hidden_size: Number of recurrent units.
            output_size: Dimensionality of the readout.
            input_plastic: Whether input weights are trainable.
            hidden_plastic: Whether recurrent weights are trainable.
            output_plastic: Whether readout weights are trainable.
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.nonlinearity = nn.ReLU()

        self.input2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

        self.input2h.requires_grad_(input_plastic)
        self.h2h.requires_grad_(hidden_plastic)
        self.h2o.requires_grad_(output_plastic)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialise linear layer weights with Kaiming normal, biases to zero."""
        for layer in [self.input2h, self.h2h, self.h2o]:
            nn.init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def init_hidden(self, batch_size, device):
        """Return a zero initial hidden state.

        Args:
            batch_size: Number of sequences in the batch.
            device: Target device for the tensor.

        Returns:
            Tensor of shape (batch_size, hidden_size) filled with zeros.
        """
        return torch.zeros(batch_size, self.hidden_size, device=device)

    def recurrence(self, x_t, h_prev):
        """Compute one step of the recurrent dynamics.

        Args:
            x_t: Input at the current time step, shape (batch, input_size).
            h_prev: Hidden state from the previous time step, shape (batch, hidden_size).

        Returns:
            h_t: Updated hidden state, shape (batch, hidden_size).
        """
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
            hidden_all: Hidden states tensor of shape (batch, time_steps, hidden_size).
        """
        if hidden is None:
            hidden = self.init_hidden(x.shape[0], x.device)

        hidden_all = []
        for t in range(x.size(1)):
            hidden = self.recurrence(x[:, t, :], hidden)
            hidden_all.append(hidden)

        hidden_all = torch.stack(hidden_all, dim=1)  # (batch, time, hidden)
        output = self.h2o(hidden_all)

        return output, hidden_all


class LeakyRNN(RNN):

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        alpha,
        input_plastic=True,
        hidden_plastic=True,
        output_plastic=True,
    ):
        """Initialise a leaky (continuous-time) Elman RNN.

        The hidden state update follows:
            h_t = (1 - alpha) * h_{t-1} + alpha * phi(W_in x_t + W_rec h_{t-1})

        Args:
            input_size: Dimensionality of the input at each time step.
            hidden_size: Number of recurrent units.
            output_size: Dimensionality of the readout.
            alpha: Leak rate in (0, 1]; equivalent to dt/tau.
            input_plastic: Whether input weights are trainable.
            hidden_plastic: Whether recurrent weights are trainable.
            output_plastic: Whether readout weights are trainable.
        """
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            input_plastic=input_plastic,
            hidden_plastic=hidden_plastic,
            output_plastic=output_plastic,
        )

        self.alpha = alpha

    def recurrence(self, x_t, h_prev):
        """Compute one leaky-integration step.

        Args:
            x_t: Input at the current time step, shape (batch, input_size).
            h_prev: Hidden state from the previous time step, shape (batch, hidden_size).

        Returns:
            h_t: Updated hidden state, shape (batch, hidden_size).
        """
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
        lr_drive = (h_prev @ self.n) @ self.m.T / self.hidden_size
        if self.J0 is None:
            recurrent = lr_drive
        else:
            recurrent = lr_drive + h_prev @ self.J0.T
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
        alpha,
        gain=0.0,
        input_plastic=True,
        hidden_plastic=True,
        output_plastic=True,
    ):
        """Initialise a low-rank leaky RNN.

        Args:
            input_size: Dimensionality of the input at each time step.
            hidden_size: Number of recurrent units.
            output_size: Dimensionality of the readout.
            rank: Rank R of the recurrent weight matrix.
            alpha: Leak rate in (0, 1]; equivalent to dt/tau.
            gain: Scaling factor for the fixed random component J_0; each
                entry is drawn from N(0, gain² / hidden_size).  gain=0
                disables the random component entirely.
            input_plastic: Whether input weights are trainable.
            hidden_plastic: Whether low-rank factors (m, n) are trainable.
            output_plastic: Whether readout weights are trainable.
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
        )

        self.alpha = alpha

    def recurrence(self, x_t, h_prev):
        """Compute one leaky step with low-rank recurrent dynamics.

        Args:
            x_t: Input at the current time step, shape (batch, input_size).
            h_prev: Hidden state from the previous time step, shape (batch, hidden_size).

        Returns:
            h_t: Updated hidden state, shape (batch, hidden_size).
        """
        lr_drive = (h_prev @ self.n) @ self.m.T / self.hidden_size
        if self.J0 is None:
            recurrent = lr_drive
        else:
            recurrent = lr_drive + h_prev @ self.J0.T
        h_new = self.nonlinearity(self.input2h(x_t) + recurrent)
        h_t = (1 - self.alpha) * h_prev + self.alpha * h_new
        return h_t


class ActorCritic(nn.Module):
    """Actor-critic wrapper around any backbone RNN.

    Takes a pre-constructed backbone (any subclass of :class:`RNN`) and adds
    separate linear policy and value heads that read from the hidden states.
    The backbone's own ``h2o`` readout layer is unused.

    Args:
        backbone: A constructed RNN instance (e.g. LeakyRNN, LowRankRNN).
        num_actions: Number of discrete actions for the policy head.
    """

    def __init__(self, backbone: RNN, num_actions: int):
        super().__init__()
        self.backbone = backbone
        self.policy_head = nn.Linear(backbone.hidden_size, num_actions)
        self.value_head = nn.Linear(backbone.hidden_size, 1)

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
        logits = self.policy_head(hidden_all)
        values = self.value_head(hidden_all).squeeze(-1)
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
        logits = self.policy_head(hidden)
        value = self.value_head(hidden).squeeze(-1)
        return logits, value, hidden
