# Neuron typing generalization artifacts

This directory is the default machine-local artifact root. Generated contents are ignored by Git.

Dataset preparation writes pinned non-image metadata under `sources/` and only frozen-manifest-selected images under
`images/`. Each image directory contains `image_manifest.parquet` and `materialization_status.json` for integrity and
download-boundary auditing. These generated files must not be committed.

Each initialized run is written to:

```text
runs/<study>/<model>__<dataset>__seed<seed>__<config-hash>/
```

Set `NEURON_TYPING_ARTIFACT_ROOT` to redirect large artifacts to another disk without changing committed configs.
