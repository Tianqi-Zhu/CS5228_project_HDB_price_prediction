[Chinese version](mlp_prediction_report.zh-CN.md)

# MLP ensemble predictions and results

This branch delivers MLP, TabM and blended predictions, cleaning controls, prediction intervals and Kaggle submissions from full-data MLP retraining. Raw CSVs are unchanged; new code, compact experiment records and directly inspectable predictions are bundled.

**Latest Kaggle screenshot scores: full-data cleaned MLP 437.34455; full-data uncleaned MLP 439.38193.** The user supplied these results on 2026-11-05; the screenshot does not identify public/private partitions and was not independently checked by API. Full-data cleaned MLP is recommended only as the lowest displayed score among the four MLP submissions below.

Under the original independent interval-calibration protocol, uncleaned blend validation RMSE is 467.03 and cleaned pure MLP scores 471.26. Historical best development RMSE 464.21 comes from another neural blend fitted through 2024-03. Kaggle displayed scores, old chronological validation and full-data runs are identified separately below.

## Kaggle submissions and full-data retraining (2026-11-05)

| Submission variant | Fitting cutoff | Actual fitting rows | Displayed Kaggle score |
| --- | --- | ---: | ---: |
| Frozen model, uncleaned MLP | 2023-12 | 105,837 | 461.64312 |
| Frozen model, obvious-low-rent cleaned MLP | 2023-12 | 104,505 | 458.91884 |
| Full-data retraining, uncleaned MLP | 2025-03 | 150,000 | 439.38193 |
| **Full-data retraining, obvious-low-rent cleaned MLP** | **2025-03** | **147,983** | **437.34455** |

Full-data retraining lowers displayed scores by 22.26119 uncleaned and 21.57429 cleaned (about 4.70%); full-data cleaned beats full-data uncleaned by 2.03738. Architecture, AdamW, seeds 42/43/44 and three epochs per network are unchanged. Fitting data, history statistics, preprocessing and batch counts change, so the gain belongs to the full refit pipeline and cannot be attributed solely to sample size. The time index retains its original quantile encoding; no new temporal representation, contemporaneous history or architecture was added.

The 44,163 labels formerly used for 2024-01 through 2024-03 calibration and 2024-04 through 2025-03 validation now enter final fitting. Reapplying cleaning to 150,000 records removes 2,017 and retains 147,983, including 43,478 from 2024-01 through 2025-03. All 50,000 test records remain; training/prediction never reads their true rents.

The test set's 380 March 2025 rows overlap the latest fitting month. Supervised fitting includes that month's known training labels, but query histories and market baselines use only pre-March data. This subset is not a fully prospective evaluation. All six reloaded MLPs reproduce predictions exactly (maximum difference 0); every CSV is checked for 50,000 rows, `Id,Predicted` names, ordered IDs, finite predictions and three-seed averaging.

No new independent local test is retained after full-data fitting, so former validation months cannot supply independent validation RMSE. External scores above come from user screenshots; `test_rmse_sgd: null` in the run summary means hidden labels are unavailable locally. Screenshot scores are separately recorded in [kaggle_results.json](../reproduction/mlp/artifacts/fullfit/kaggle_results.json).

- [Full-data cleaned MLP submission CSV](../reproduction/mlp/artifacts/fullfit/submission_mlp_fullfit_cleaned.csv)
- [Full-data uncleaned MLP submission CSV](../reproduction/mlp/artifacts/fullfit/submission_mlp_fullfit_uncleaned.csv)
- [Training summary](../reproduction/mlp/artifacts/fullfit/summary.json), [row checks](../reproduction/mlp/artifacts/fullfit/verification.json), [removals](../reproduction/mlp/artifacts/fullfit/obvious_low/removed_training_rows.csv)
- [Full-data score screenshot](../reproduction/mlp/artifacts/fullfit/evidence/kaggle_fullfit_2026-11-05.png), [original frozen-model screenshot](../reproduction/mlp/artifacts/fullfit/evidence/kaggle_frozen_2026-11-05.png)

Full-data CSVs provide point predictions without the old 90%/95% intervals. Old coverage results do not transfer to the full-data models.

## Temporal decay weights: local comparison

Optional `--half-life-months` sets raw row weights to `2 ** (-months since the latest fitting month / half-life)`, then normalizes their fitting-set mean to 1. Regression and auxiliary rent-band classification share the weights. Only the loss changes; property inputs, history definitions and quantile preprocessing remain fixed. Omitting the option preserves uniform weights.

With obvious-low-rent cleaning, three epochs and seeds 42/43/44, compare uniform weights with half-lives 6/12/24/36 months. First evaluate disjoint earlier windows 2023-04 through 2023-09 and 2023-10 through 2024-03, retaining every row and using unweighted metrics. Pooled RMSEs are 577.6236/578.5251/578.1931/577.8959/577.7108: uniform is best overall, 36 months best among positive half-lives. Lock that weighted candidate before the later annual evaluation.

| Annual validation: 2024-04 through 2025-03, 35,158 rows | RMSE | MAE |
| --- | ---: | ---: |
| Uniform weights | 470.2343 | 341.4548 |
| 36-month half-life | 469.7138 | 341.4755 |

Both fit 113,324 cleaned records through 2024-03. Annual RMSE improves by 0.5205, but MAE barely changes and the earlier pooled comparison loses; this is insufficient for a stable-gain claim. Fixing three epochs isolates weighting, without reselecting duration per half-life. All local windows were previously inspected and remain development comparisons.

At a 36-month half-life, raw weights one, two and three years earlier are about 0.794/0.630/0.500 relative to the latest month. A full-data weighted candidate was trained for Kaggle comparison; 437.34455 still belongs to the original uniformly weighted cleaned refit. Full-data training and search commands are:

```bash
python -m analysis.main mlp fullfit --half-life-months 36 --output-dir reports/local_runs/mlp_weighted_h36
python -m analysis.main mlp recency-search --output-dir reports/local_runs/mlp_recency_search
```

## Inputs and networks

Inputs include type, model, town, street/block IDs, area, time, age, coordinates, building attributes, amenity distances/counts and strictly historical rent statistics. Street/block categories use learned embeddings; categories with fewer than four records map to an unknown/rare bucket. Numerical imputation and quantile transforms fit the current fitting partition only.

The standalone MLP concatenates embeddings and numerical inputs, then uses a width-128 projection and two 128-unit dense blocks with SiLU, BatchNorm/LayerNorm and dropout. It predicts the residual from a market-adjusted historical baseline, then restores SGD rent; an auxiliary price-band classifier trains without reading true bands at inference. AdamW uses learning rate 0.0005 and batch 1,024.

TabM uses official `tabm==0.0.3`, categorical and piecewise-linear numerical embeddings, width 128, two blocks and 16 parameter-sharing members per network. AdamW uses learning rate 0.001 and batch 512. Standardized-rent outputs are restored to SGD.

Deep Ensemble trains three independent MLPs with seeds 42/43/44; TabM also has three independent seeded refits. Average each TabM's internal members, then its three seeds. The final blend gives MLP and TabM ensembles 50% each. Shared-parameter members are not 48 independent networks.

The main ensemble excludes RNN/LSTM latent vectors. Earlier depth/latent experiments remain historical controls; not every attempted change improved results.

## Original interval experiment: chronological split

| Stage | Data | Purpose |
| --- | --- | --- |
| Epoch selection | Fit 2021-01 through 2023-09; validate 2023-10 through 2023-12 | Select three MLP epochs and four TabM epochs |
| Refit independent networks | 2021-01 through 2023-12, 105,837 rows | Three seeds; refit all training preprocessing |
| Interval calibration | 2024-01 through 2024-03, 9,005 rows | Calibrate intervals only; do not reselect networks or blend weights |
| Full validation | 2024-04 through 2025-03, 35,158 rows | Compare RMSE, MAE, bias and coverage |
| Test predictions | All 50,000 rows of `test.csv` | Preserve sample-submission ID order; true labels are unavailable for scoring |

The original six interval CSVs use models frozen through 2023-12 to preserve calibration. Validation spans horizons 4-15 and test spans 15-31; longer predictions/intervals extrapolate beyond validated horizons. Two new full-data submissions above are separate, with fitting cutoff advanced to 2025-03.

## Original frozen-model local results

| Method | RMSE | MAE | Mean bias: predicted minus actual |
| --- | ---: | ---: | ---: |
| Uncleaned MLP ensemble | 471.80 | 351.01 | -34.52 |
| Uncleaned TabM ensemble | 469.74 | 342.92 | +1.99 |
| Uncleaned blend | 467.03 | 343.66 | -16.26 |
| Blend with historical-risk correction candidate | 465.34 | 344.72 | -31.25 |
| MLP after obvious-low-rent cleaning | 471.26 | 345.46 | -7.61 |
| Blend after obvious-low-rent cleaning | 468.79 | 337.00 | +40.44 |
| Blend after strict-high-rent cleaning | 476.25 | 339.09 | +78.28 |

Historical-risk correction learns small adjustments from earlier errors. It lowers RMSE but raises MAE and fails the prespecified two-metric replacement rule. Failed alternatives, including shrinkage toward historical low-rent means, are also retained.

See all nine cleaning/model rows in [cleaning metrics](https://github.com/Tianqi-Zhu/CS5228_project_HDB_price_prediction/blob/322fa4c20d83f20e19328823203873a0f81e0f4a/reports/experiments/mlp_summary/cleaning/comparison_metrics.csv). Cleaning raises predicted rents, benefiting high-rent groups while worsening low-rent errors: [grouped results](https://github.com/Tianqi-Zhu/CS5228_project_HDB_price_prediction/blob/322fa4c20d83f20e19328823203873a0f81e0f4a/reports/experiments/mlp_summary/cleaning/rent_band_metrics.csv).

The historical best point blend has **RMSE 464.21, MAE 336.94**, fitting 114,842 rows from 2021-01 through 2024-03 and evaluating the same 35,158 outer records. Training size, features and model composition all change, and the outer months were repeatedly inspected; the gain cannot be attributed solely to network depth. Its [row-level validation predictions](https://github.com/Tianqi-Zhu/CS5228_project_HDB_price_prediction/blob/322fa4c20d83f20e19328823203873a0f81e0f4a/reports/experiments/mlp_summary/validation/best_point_blend_outer.csv) are bundled, without corresponding test-interval files.

Plain one-hot MLPs with six/eight layers and no latent inputs score 488.35/484.79; RNN/LSTM additions do not consistently help. Learned embeddings, histories, residual targets, numerical representations and blending motivate later work. See positive and negative results in the [historical catalog](https://github.com/Tianqi-Zhu/CS5228_project_HDB_price_prediction/blob/322fa4c20d83f20e19328823203873a0f81e0f4a/reports/experiments/mlp_summary/historical_catalog.csv). Early broken numerical-embedding runs are explicitly marked and are not valid piecewise-embedding comparisons.

## Interpreting confidence

Outputs are prediction intervals for individual rents, not probabilities that predictions are correct. Independent-network standard deviation measures disagreement, not total prediction error; it is not divided by the square root of the seed count.

Calibration uses `abs(actual rent - predicted rent) / max(model-disagreement standard deviation, 100 SGD)`. Finite-sample corrected rank quantiles on independent calibration data produce 90%/95% intervals. Fixed-width absolute-error calibration is also retained; outer performance does not select the method. See [Deep Ensembles](https://proceedings.neurips.cc/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html) and [Conformal Prediction](https://arxiv.org/abs/2107.07511).

| Uncleaned model | Empirical 90% coverage | Empirical 95% coverage | Mean 90% interval width |
| --- | ---: | ---: | ---: |
| MLP | 92.56% | 96.28% | S$1,655 |
| TabM | 92.73% | 96.21% | S$1,666 |
| Blend | 92.79% | 96.31% | S$1,670 |

These are empirical development-window coverages. Temporal change, correlated same-block transactions and changing composition violate simple IID assumptions; future months or each flat type need not reach nominal coverage. Low-rent blend coverage is only 29.4% at 90% nominal, despite aggregate coverage above 90%.

## Where remaining errors concentrate

In the latest risk-corrected candidate, actual rents≤1,900 comprise 3.7% of rows but 25.7% of squared error, with mean overprediction about S$1,112; rents>3,800 are underpredicted by about S$427. The worst 5% contribute 40.9% of squared error. Executive, maisonette, large flats and Jurong West appear more often among large errors. These overlapping groups guide investigation, not causal explanation.

Large errors are not mainly unseen properties: median history support is nine among the worst 5%, versus ten overall. Disagreement also poorly flags difficulty: of 1,422 errors above 1,000, 1,182 have disagreement standard deviation below 100.

Multiple labels for identical inputs create local limits. A validation group with rents 4,000/5,000/6,000 has minimum within-group RMSE 816.50 at a constant prediction of 5,000; the uncleaned blend predicts 5,028.49, scoring 816.99, close to that local optimum. This concerns three cases only, not a proof that overall RMSE cannot reach 450. [Case and calculations](https://github.com/Tianqi-Zhu/CS5228_project_HDB_price_prediction/blob/322fa4c20d83f20e19328823203873a0f81e0f4a/reports/experiments/mlp_summary/uncertainty_diagnostics/indistinguishable_inputs_proof.json)

## Original interval prediction files

Each file has 50,000 rows in sample-submission ID order. Cleaning/risk-correction variants are retained as candidates without overwriting original predictions.

| File | Model and purpose |
| --- | --- |
| [Uncleaned MLP](https://github.com/Tianqi-Zhu/CS5228_project_HDB_price_prediction/blob/322fa4c20d83f20e19328823203873a0f81e0f4a/reports/experiments/mlp_summary/predictions/uncleaned_mlp_test_intervals.csv) | Independent MLP control; validation RMSE 471.80 |
| [Uncleaned blend](https://github.com/Tianqi-Zhu/CS5228_project_HDB_price_prediction/blob/322fa4c20d83f20e19328823203873a0f81e0f4a/reports/experiments/mlp_summary/predictions/uncleaned_hybrid_test_intervals.csv) | Main blend control; validation RMSE 467.03 |
| [Obvious-low-rent cleaned MLP](https://github.com/Tianqi-Zhu/CS5228_project_HDB_price_prediction/blob/322fa4c20d83f20e19328823203873a0f81e0f4a/reports/experiments/mlp_summary/predictions/obvious_low_mlp_test_intervals.csv) | Lowest pure-MLP error among this round's cleaning variants; validation RMSE 471.26 |
| [Obvious-low-rent cleaned blend](https://github.com/Tianqi-Zhu/CS5228_project_HDB_price_prediction/blob/322fa4c20d83f20e19328823203873a0f81e0f4a/reports/experiments/mlp_summary/predictions/obvious_low_hybrid_test_intervals.csv) | MAE 337.00; validation RMSE 468.79 |
| [Strict-high-rent cleaned blend](https://github.com/Tianqi-Zhu/CS5228_project_HDB_price_prediction/blob/322fa4c20d83f20e19328823203873a0f81e0f4a/reports/experiments/mlp_summary/predictions/highest_only_hybrid_test_intervals.csv) | Failed control; validation RMSE 476.25 |
| [Historical-risk-corrected blend](https://github.com/Tianqi-Zhu/CS5228_project_HDB_price_prediction/blob/322fa4c20d83f20e19328823203873a0f81e0f4a/reports/experiments/mlp_summary/predictions/risk_corrected_hybrid_test_intervals.csv) | Candidate with RMSE 465.34 but slightly higher MAE |

`Id` is the competition sample ID; `month` is approval month; `prediction_sgd` is predicted monthly rent; `ensemble_std_sgd` is disagreement; `pi90_lower_sgd/upper_sgd` and `pi95_lower_sgd/upper_sgd` are interval bounds. Hidden test labels are unavailable locally, so test RMSE cannot be computed; see the first section for screenshot scores.

To create competition format, take `Id` and `prediction_sgd` from any prediction file and rename the latter `Predicted`. Interval files contain extra fields and are not themselves two-column submissions.

## Reproduction and verification

Main experiments ran on Python 3.12, CPU, PyTorch 2.14.1, TabM 0.0.3 and rtdl-num-embeddings 0.0.12. Training uses two Torch threads without CUDA. Use new output directories to preserve completed runs.

All 37 relevant pre-release tests passed. Archive verification also recomputes RMSE for five row-level validation files and checks IDs, months and interval order across six 50,000-row test files. Numerical verification is not an independent future test.

```bash
python -m pip install -r requirements/mlp.txt
python -m analysis.main mlp verify
python -m analysis.main mlp test

# Regenerate uncleaned baseline, calibration and test predictions
python -m analysis.main mlp intervals --output-dir reports/local_runs/mlp_uncertainty --predict-test

# Reuse the generated baseline settings for both cleaning experiments
python -m analysis.main mlp cleaning --baseline-dir reports/local_runs/mlp_uncertainty --output-dir reports/local_runs/mlp_cleaning

# Refit full-data MLP with archived original settings/epochs and export two Kaggle-format CSVs
python -m analysis.main mlp fullfit --baseline-dir reproduction/mlp/config/interval_baseline --output-dir reports/local_runs/mlp_fullfit
```

On macOS, OpenMP library issues may require `DYLD_FALLBACK_LIBRARY_PATH=/opt/miniconda3/lib`; headless environments can set `MPLBACKEND=Agg` and `MPLCONFIGDIR=/tmp/hdb-mpl`. Hardware/numerical-library differences may cause small changes. `--resume` is accepted only when settings, data and code hashes match.

The [delivery inventory](https://github.com/Tianqi-Zhu/CS5228_project_HDB_price_prediction/blob/322fa4c20d83f20e19328823203873a0f81e0f4a/reports/experiments/mlp_summary/delivery.json) distinguishes historical point results from test files; the [manifest](https://github.com/Tianqi-Zhu/CS5228_project_HDB_price_prediction/blob/322fa4c20d83f20e19328823203873a0f81e0f4a/reports/experiments/mlp_summary/manifest.json) records compact-evidence SHA-256 hashes. Verification recomputes archived validation RMSE and checks test IDs, months and interval order without downloading models. Weights/preprocessing caches remain local and outside this branch; retraining regenerates them.
