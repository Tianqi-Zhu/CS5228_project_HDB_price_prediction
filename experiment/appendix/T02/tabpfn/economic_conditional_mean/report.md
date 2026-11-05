# Temporal economic conditioning of expected rent

Jointly trained property FT-Transformer plus an eight-unit LSTM over twelve historical months. A mean head predicts a rent-level shift and eight property-dependent temporal loadings. Prediction is the original causal local baseline plus the property residual and temporal conditional correction; no future observed mean is supplied.

Matched rent-only and rent+COE designs differ only in the available COE channels. Stocks are masked. Both start with identical property weights and a zero temporal correction. All parameters train with standardized individual-rental MSE plus0.01 times squared correction, using the original chronological folds and extra month of macro lag.

## Earlier chronological comparisons

| Window | Design | RMSE (SGD) | Change vs T0 | Change vs matched rent-only | Best epoch |
| --- | --- | ---: | ---: | ---: | ---: |
| early_2022 | rent | 640.510 | -8.094 | +0.000 | 14 |
| early_2022 | rent_coe | 648.691 | +0.087 | +8.181 | 3 |
| early_2023 | rent | 580.578 | +15.145 | +0.000 | 3 |
| early_2023 | rent_coe | 571.107 | +5.673 | -9.471 | 6 |

Earlier gate passed: **False**. It requires the COE model to beat matched rent-only and original T0 in both earlier windows. Refit epochs selected earlier: {"rent": 9, "rent_coe": 5}.

The declared earlier gate stopped this candidate. No later model fits, later scoring, or additional architecture search was performed.

## Limits and reproduction

Only51 labeled months exist. Earlier windows each contain2000 sampled queries with population weights, not2000 independent economic observations. Earlier validation selects checkpoints and the gate, so these are selection results rather than independent tests. Later periods have already been repeatedly examined in the project. Existing baseline histories, code, checkpoints and submission CSVs remain unchanged.

Run `.venv/Scripts/python.exe analysis/run_economic_conditional_mean.py --stage all --device cuda`. CPU and CUDA smoke use `--stage smoke --device cpu` and `--stage smoke --device cuda`. Settings record sources/hashes; per-run artifacts contain learning curves, conditional-shift predictions, exact checkpoint replay, timing and peak VRAM. Worker resumes completed runs and per-epoch optimizer/RNG state.
