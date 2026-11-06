[Chinese version](README.zh-CN.md)

# Environment and optional extensions

Core dependencies are maintained in root `environment.yml`, covering analysis, conventional models, reporting and final tree/Transformer blending.

## Main environment

Run from the repository root:

```powershell
conda env create -f environment.yml
conda activate cs5228
```

This file uses the verified Windows/NVIDIA configuration, pinning PyTorch 2.11.0+cu128, RTDL and numeric-embedding training versions; plotting, PDF and Excel tools retain moderate version ranges.
Versions come from the executed local environment and isolated dependencies; see audit records. This change does not install or update the current environment.
Other hardware requires a compatible PyTorch build.

Run `python -m analysis.main preflight --profile core` to inspect core versions and metadata locations;
`--profile wave06` and `--profile wave07` check optional routes. Preflight neither installs nor imports models; see the [running guide](../docs/guides/running_experiments.md) for boundaries.

## Optional extensions

| File | Purpose | Installation location |
| --- | --- | --- |
| [optional_deep_models.txt](optional_deep_models.txt) | Wave-six TabM and RealMLP; numeric embeddings are already in the main environment | `.cache/deep-deps/` |
| [optional_pretrained_transformers.txt](optional_pretrained_transformers.txt) | Wave-seven TabDPT, Nori and related support packages | `.cache/transformer-deps/` |

Only when reproducing these routes, activate the main environment and run:

```powershell
python -m pip install --no-deps --target .cache/deep-deps -r requirements/optional_deep_models.txt
python -m pip install --no-deps --target .cache/transformer-deps -r requirements/optional_pretrained_transformers.txt
```

Optional experiments retain the author-source, public-weight and existing `.cache/foundation-deps/` shared-support setup,
see the [running guide](../docs/guides/running_experiments.md) and [wave-by-wave experiments](../experiment/04_optimization_history/README.md).
Optional TXT files support isolated route-specific installation; source code, weights and shared support packages remain prerequisites.

Old profiles/extras manifests and root requirements.txt were removed according to the audit draft;
environment.yml is authoritative for core versions, avoiding drifting base-installation entry points.
