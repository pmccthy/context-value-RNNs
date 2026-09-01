# cross_model_vs_experiment/

New analyses that combine the RNN model results with the real experimental
data in `~/Documents/neuronal-representations/results/transfer`.

## Chi-squared model-fit test (done: real data now available, ready to run)

`reversal_study/code/population_similarity.py` already implements
`chi2_fit_to_data()` / `fig_group_chi2()` -- a chi-squared goodness-of-fit test
of each model type's pooled responder-group composition (the same 7-group
taxonomy as `responder_groups`/`celltype_group_grid` figures) against a
`--data-counts group_counts.json` file. **It has only ever been run against a
synthetic placeholder (`fake_data_counts()`)** -- real neural counts had never
been supplied.

`extract_experiment_group_counts.py` (this dir) fixes that: it reads the real
tidy tables in `~/Documents/neuronal-representations/results/transfer/data/subgroups/`
and writes `group_counts/{expert,reversal_pre,reversal_post}.json` in the exact
schema the test expects. Already run -- see those three files for the real
counts. Run `python3 extract_experiment_group_counts.py --help` for the
mapping logic (the reversal CSV labels stimuli by pre->post transition, not
current value, so it's re-mapped per phase).

**To generate the real (not fake) comparison figure**, in an env with scipy
(the sandbox this was drafted in has no network access to install scipy --
your local `cxval`/`neuronal_representations` env should already have it):

```
cd context-value-RNNs
python3 unified_figures/reversal/code/population_similarity.py \
    --pre unified_figures/transfer/figure_data \
    --out unified_figures/cross_model_vs_experiment/figures \
    --data-counts unified_figures/cross_model_vs_experiment/group_counts/expert.json
```

A manual preview (scipy-free, chi2 statistic only, no p-value) was run during
drafting against the three `transfer_final` models and the real `expert.json`
counts:

| model                          | N units (pooled) | chi2 vs. real data | rank |
|---------------------------------|------------------:|---------------------:|:----:|
| classif_rl ("SSL + RL")         | 2689               | 2645                 | 1 (best fit) |
| rl_only ("RL only")             | 2395               | 3965                 | 2    |
| classif_rl_readout_only         | 3145               | 9514                 | 3 (worst fit)|

Read this as a *relative* ranking, not absolute goodness of fit -- with
thousands of units, chi2 is large and p-values will be ~0 for all three
models regardless (real single-neuron data is never going to match a small
RNN's responder-group shape exactly). The classif_rl (SSL+RL) representation
is the closest of the three to the real proportions; rerun the command above
to get the proper p-values and the bar-chart figure.

For the reversal comparison, `group_counts/reversal_pre.json` and
`reversal_post.json` are ready but not yet run against the model side --
that needs `reversal_study`'s own pre/post `figure_data` directories (e.g.
under `results/reversal_5000/`), not `transfer_final`'s. Next step, not done
here.

## Planned: run the full experimental analysis pipeline on model activations

Not started. The experimental repo's `extract_*.py` scripts (population
means, responsiveness t-test, PCA, TDR, decoding -- see
`~/Documents/neuronal-representations/analysis/` and
`results/transfer/extract/`) are written against real 2-photon session
pickles. Reusing them on model activations means either (a) reshaping
`figure_data.pkl` / `activations.npz` into the same tidy long-format the
`extract_*.py` scripts expect and running those scripts unmodified, or (b)
reimplementing the same statistics directly against the model's own arrays
(closer to what `cxval/analysis.py` already does, e.g.
`responsive_proportions_ttest`, `compute_unit_tuning`). Worth deciding which
before starting -- see FIGURE_LIST.md, section D3.

## Planned: population-activity overlay figure

Not started. Plot the model's `population_activity_per_stim` trace next to the
experimental `draw_expert_mean` / `draw_reversal_mean` trace (same axes,
`style.py` colors/linestyles) as a direct visual comparison -- doesn't exist
in either repo yet. See FIGURE_LIST.md, section D2.
