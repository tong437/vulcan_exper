# Reports

Commit compact, reviewable conclusions here: Markdown reports, final tables, and publication figures. Large raw
outputs and intermediate plots stay under `artifacts/neuron_typing_generalization/`.

The frozen VQAv2 pilot closure produces `vqav2_pilot_results.md`, its machine-readable JSON companion, three compact
cross-model figures, and `vqav2_direct_streaming_validation.json`. Per-model diagnostic figures remain in each ignored
run directory.

The four-dataset synthesis produces `main_synthesis_results.md` and JSON, `main_results_table.csv`,
`paired_model_contrasts.csv`, normalized-depth curves, and a forest plot of Bonferroni familywise 95% intervals. Model
contrasts are paired only within a dataset; VizWiz is descriptive only.
