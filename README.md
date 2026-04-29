# context-value-RNNs

An investigation into how value representations emerge in recurrent neural networks trained on context-dependent reinforcement learning tasks.

## Overview

How does the brain represent the value of stimuli when those values depend on context? This repository provides tools to train and analyse recurrent neural networks (RNNs) on a context-dependent bandit task, with the goal of understanding what representational geometry value information adopts in recurrent circuits.

The task structure is inspired by multi-context reward learning paradigms: a set of stimuli each has a base reward probability, which is then perturbed by context-specific noise. Networks must learn to track stimulus values across context switches, providing a tractable model system for studying contextual value coding, credit assignment, and the geometry of value representations in RNN hidden states.

The core design separates task generation, model architecture, and analysis. Task data (value matrices, stimulus sequences, state time-series) can be generated independently of the model, making it straightforward to swap in different RNN architectures or training objectives.

## Structure

```
cxval/              # Installable package
    tasks.py        # Task generation: ValueMatrix, StimulusSequence, StateSequence
    models.py       # RNN model architectures (BackboneRNN, actor-critic variants)
    agents.py       # (reserved for agent training logic)
    analysis.py     # (reserved for representational analysis)
    vis.py          # Visualisation utilities
nb/                 # Jupyter notebooks for experiments
scripts/            # Standalone training and analysis scripts
task_data/          # Pre-generated task datasets
results/            # Saved model outputs and figures (not tracked by git)
```

## Installation

Requires Python ≥ 3.11. Install into a mamba/conda environment:

```bash
mamba activate cxval
pip install -e .
```

## Usage

```python
from cxval.tasks import ValueMatrix, StimulusSequence, StateSequence

# Define the task structure
vm = ValueMatrix(
    seed=0,
    contexts=list(range(4)),
    stimuli=list(range(6)),
    delta_context=0.2,
)
vm.generate_base_values(seed=0)
value_matrix = vm.generate_value_matrix(seed=1)

# Generate a trial sequence
seq = StimulusSequence(
    value_matrix=value_matrix,
    trials_per_phase=30,
    phases_per_context=2,
    context_order='random',
)
trial_contexts, trial_stimuli = seq.generate(seed=42)

# Build the full state time-series for RNN input
states_gen = StateSequence(
    stimulus_sequence=seq,
    value_matrix=value_matrix,
    stim_timesteps=5,
    reward_timesteps=2,
    iti_timesteps=(3, 8),
)
states, rewards = states_gen.generate(seed=42)
```

## Dependencies

- [PyTorch](https://pytorch.org/) — RNN model training
- NumPy, SciPy, Matplotlib
