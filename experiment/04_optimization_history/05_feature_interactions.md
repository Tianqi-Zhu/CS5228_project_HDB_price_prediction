[Chinese version](05_feature_interactions.zh-CN.md)

# Round 5:controlled model × feature comparisons

Focus on models/features as requested, pause new parameter search. Three representative learners×six inputs×three windows=54 fits, with settings, durations, seeds fixed within learners.

## 1. Established feature gains

Fixed-duration LightGBM annual RMSE over 19→27→47→62 inputs: 481.60→467.80→466.97→457.84; GPU CatBoost479.33→466.43→465.99→455.17. Substantial feature gains remain real despite small recent tuning increments.

## 2. New feature hypotheses

| Feature group | Added fields | Hypothesis/provenance |
|---|---:|---|
| Historical distribution | 60 | Block/type and street/type full/latest12-month Q10/Q25/median/Q75/Q90, IQR, low shares, low/high component means, modal-bin shares, entropy, time since low rent, last-month median and recent 6-month/earlier median difference. Normalize by observed source-month/type mean. |
| Building structure | 12 | Supplied-building counts within300m/800m/1.5km; nearest16 height/completion-year distributions, high-rise/old shares and own-building contrasts. No rental labels. |

Pruning controls:`no36` removes five36-month fields, leaving57;`only24` also removes five12-month fields, leaving52. Ordinary/market-relative histories remain.

Distribution histories use rents strictly before each information-cutoff month. Future queries freeze statistics; only time-since-low-rent grows. Unknown groups have count0/missing statistics. Building data are a competition snapshot, not verified historical inventories.

Raw audit: FURNISHED is constant yes, FEE constant0; area/model are unique within block/type. Readding them to that key cannot distinguish tenants/units. No unit-floor, renovation or renewal IDs exist.

## 3. Model settings

| Model | Fixed configuration |
|---|---|
| LightGBM | 15 leaves, 353 trees, learning rate0.03, L2=10, min leaf40 |
| GPU CatBoost | Depth6,535 iterations, learning rate0.04, L2=10 |
| GPU XGBoost | Depth3,1046 iterations, learning rate0.03, L2=20, min child weight20 |

Durations come from earlier experiments; no validation-specific stopping/retuning. Six inputs within each model share rows; normal categorical handling may differ by learner. GPU CatBoost has small numerical nondeterminism; sub-SGD changes need new seeds/windows.

## 4. Main controls

### 2024-04–2025-03:35,158 rows

| Model | Original62 inputs | + Historical distribution | + Building structure | Both additions | Remove36-month fields | Retain24-month anniversary only |
|---|---:|---:|---:|---:|---:|---:|
| LightGBM | 457.8450 | 457.0086 | 457.5161 | 456.7527 | 457.8827 | 459.3434 |
| CatBoost | 455.2039 | 455.4363 | 455.0816 | 455.2448 | 455.5086 | 456.4758 |
| XGBoost | 455.8482 | 455.8993 | 455.6901 | 455.3031 | 456.2694 | 457.1674 |

### 2024-06–2025-03:28,995 rows

| Model | Original62 inputs | + Historical distribution | + Building structure | Both additions | Remove36-month fields | Retain24-month anniversary only |
|---|---:|---:|---:|---:|---:|---:|
| LightGBM | 447.1701 | 447.2156 | 447.1111 | 446.9913 | 447.3701 | 448.6101 |
| CatBoost | 446.1608 | 446.2064 | 446.0444 | 446.1903 | 446.0275 | 447.3196 |
| XGBoost | 445.6576 | 445.8572 | 446.0440 | 445.9626 | 445.8147 | 447.5220 |

### Earlier2023-10–2024-03:17,977 rows

| Model | Original62 inputs | + Historical distribution | + Building structure | Both additions | Remove36-month fields | Retain24-month anniversary only |
|---|---:|---:|---:|---:|---:|---:|
| LightGBM | 527.6640 | 528.3354 | 527.7810 | 528.3203 | 527.6640 | 529.5079 |
| CatBoost | 526.1404 | 526.1294 | 526.2390 | 526.2576 | 526.5394 | 527.8272 |
| XGBoost | 525.9392 | 526.6298 | 526.0118 | 526.1097 | 526.2006 | 527.7293 |

### Findings this round

- Historical distributions alone improve neither later window consistently for any learner; unobserved contract differences remain.
- Building structure slightly improves LightGBM/CatBoost consistently, but worsens recent XGBoost.
- Combined groups improve both later windows only for LightGBM; XGBoost annual gains do not transfer recently.
- Keeping24-month cohorts alone worsens all three models/windows; retain12/24/36-month combination.
- Insufficient evidence replaces442.90 blend. Main contribution is controlled model/feature interaction, not a new minimum.

## 5. Interpretation beyond minimum scores

Delta RMSE=augmented minus original 62-input RMSE; negative improves. Early/annual/recent differences:`feature_stability.csv`. Overlapping later windows mean2/2 directional agreement, not two independent confirmations.

Distribution estimates are unstable with sparse support; modal shares/entropy depend on bins. Existing coordinates, age/amenities may replace building information. Added/deleted fields alter random column sampling; one fixed seed does not remove randomness. Check small changes with more seeds/new windows.

These standalone feature controls are not matched architecture comparisons against the heterogeneous442.90 blend. A chance later minimum does not replace submissions; target 430 needs complete common-window validation.

## 6. Verification

Verify54 row files against raw IDs/months/labels/counts/metrics; 36 later models reload exactly; raw hashes unchanged; 29 tests pass. Retain all validation rows, no target-month rent features/public rent answers.

Tables:`model_feature_matrix.csv`; stability:`feature_stability.csv`; plot:`model_feature_interaction.png`; field audit:`source_audit.json`; checks:`verification.json`.
