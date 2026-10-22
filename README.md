# context-value-RNNs

Recurrent neural network (RNN) models of context-dependent value coding, built in **PyTorch**.

We train RNNs on context-dependent reward tasks by combining **reinforcement learning** (an actor-critic objective) with **self-supervised learning** (auxiliary prediction losses on the same hidden state, for example stimulus or context identity). The aim is to **recapitulate experimentally observed neural representations and behaviour**: the representational geometry of value and context in the hidden units, and behavioural readouts such as lick vigour and adaptation after reversals. We then ask which training objectives and architectures produce brain-like solutions.

## Repository layout

- `cxval/` is the core package:
  - `tasks.py`, `envs.py`: task definitions (value matrices, stimulus/context sequences) and the environment
  - `models.py`: RNN backbones (vanilla, leaky, low-rank, GRU) and the actor-critic wrapper
  - `agents.py`, `vigour.py`, `context_vigour.py`, `batched.py`: training loops for RL, the auxiliary SSL heads, and batched training
  - `analysis.py`, `geometry.py`, `vis.py`: decoding, representational geometry and plotting
- `scripts/` holds training runs and parameter sweeps (for example the SSL vs. RL loss-balance sweep)
- `nb/` holds exploratory and analysis notebooks
- `docs/` holds design notes
- `tests/` holds unit tests

## Setup

Requires Python ≥ 3.11 and PyTorch.

```bash
pip install torch numpy scipy scikit-learn matplotlib
pip install -e .
```
