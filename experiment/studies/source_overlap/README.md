[Chinese version](README.zh-CN.md)

# Public rental-data overlap audit

On 2026-10-24, following a gap between scores of 172–183 and above 400 on the CS5228-2610 Kaggle Public leaderboard,
this audit checks whether competition records overlap publicly disclosed original rental records.

## Verified sources

- [Competition Public leaderboard](https://www.kaggle.com/competitions/cs-5228-2610-project/leaderboard):
  the top nine then ranged from 172.42640 to 183.18108, with tenth at 411.75928; the page described approximately 50% test coverage.
  Read through a browser the user was authorized to access; no predictions submitted or posts made.
- [Official HDB public dataset](https://data.gov.sg/datasets/d_c9f57187485a850908655db0e8cfe651/view):
  212,597 records from 2021-01 to 2026-09, including approval month, Town, Block, street, flat type and monthly rent.
  Downloaded through the public API linked on the official website; full snapshot hashes are in `results.json`.

Matching uses **month + Town + Block + normalized street + flat type**, standardizing only case, whitespace and government address
abbreviations such as ROAD/RD, NORTH/NTH, COMMONWEALTH/C'WEALTH and MARKET/MKT.
Target rents are excluded from keys; no fuzzy matching is used.

## Results

| Check | Result |
| --- | ---: |
| Training records matching public groups | 150,000 / 150,000, 100% |
| True training rent present in matching public group | 150,000 / 150,000, 100% |
| Test records matching public groups | 50,000 / 50,000, 100% |
| Test rows with a single distinct public-group rent | 36,487, 72.974% |
| Test rows with multiple distinct public-group rents | 13,513, 27.026% |

The results directly show substantial dataset overlap. They support public-record matching and disambiguation as a
possible explanation, without proving that any particular team used it.

## Comparisons on one known-label window

The window is 2024-04 to 2025-03, with 35,158 rows; known local training labels are used to investigate the mechanism.

| Method | RMSE (SGD) |
| --- | ---: |
| LightGBM trained on past information | 466.25 |
| Direct matching-public-group mean | 181.25 |
| Remove one occurrence of the row's true rent from the candidate pool, then average; fall back to LightGBM if empty | 517.88 |

Public-group means reconstruct all 150,000 training rows with RMSE 179.95.
Low reconstruction error depends on the pool already containing the sample's target. A public group with one distinct rent
returns it directly; multiple leases with different rents leave within-group error under averaging.
This can naturally produce reconstruction errors near 170–180.

The final row is only a self-label-removal diagnostic: known local labels allow subtraction of one matching-value occurrence
from the candidate multiset. It is neither a proposed competition prediction method nor an estimate of irreducible error.
Test labels are unavailable, so this audit does not measure actual Kaggle test RMSE.

Normal predictive blending achieves local RMSE 465.66; see
[model comparison](../../../docs/methods/model_comparison.md). Public-record reconstruction uses disclosed rents from target months,
so it is not a past-information generalization score and was excluded from model inputs, early stopping and blend-weight selection.

## Rule boundaries

The [competition overview](https://www.kaggle.com/competitions/cs-5228-2610-project/overview) welcomes additional data collection.
The [rules page](https://www.kaggle.com/competitions/cs-5228-2610-project/rules) provides general competition requirements and
Kaggle rules. Visible pages did not explicitly authorize direct recovery of original test-period rents from public sources; course
supplements or organizer interpretations may further constrain this use. This audit does not declare the method compliant or prohibited.
No discussion was available to cross-check at the time.

This was a local provenance audit only; no public-label-filled Kaggle submission was generated or uploaded.

## Reproduction

```powershell
# First download of official public data and audit; cache: .cache/public_rents/
python -m analysis.main source-audit --download

# Existing cache: no network required
python -m analysis.main source-audit
```

`results.json` records match rates, label agreement, reconstruction errors and raw-file hashes.
`competition_context.json` records verified competition-page facts and dates; monthly coverage is a separate CSV.
Fallback-model and self-label-removal comparisons require the existing local LightGBM validation-prediction cache.
Original `data/train.csv` and `data/test.csv` remain unchanged throughout.
