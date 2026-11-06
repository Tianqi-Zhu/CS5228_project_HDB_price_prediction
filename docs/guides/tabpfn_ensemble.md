# Regenerated tree/TabPFN ensemble submissions

The ensemble combines 50% of the fixed seven-tree anniversary ensemble with
50% of the 87-feature, frozen TabPFN-3.5-Fast model. All rows, member weights,
earlier-selected tree iterations, two TabPFN estimator views and seed 42 are
fixed. No test labels or new validation early stopping are used.

## Results and category preprocessing

Fresh validation fits use 121,005 January 2021–May 2024 rows and predict all
28,995 June 2024–March 2025 rows. Full-data fits use all 150,000 supplied
training rows and predict the 50,000 original test rows.

| Category variant | Validation RMSE (SGD) | Validation MAE (SGD) |
|---|---:|---:|
| Original categories | 440.95195750 | 318.88791651 |
| Grouped categories | 440.77070676 | 317.76248494 |

The grouped variant replaces the direct `FLAT_MODEL_CLEAN` category in both
tree and TabPFN inputs. It normalizes case/whitespace, merges `type s1` and
`type s2` into `type s`, and merges `model a maisonette`, `premium maisonette`
and `improved maisonette` into `maisonette`. Other model labels remain separate.
There are 20 original source labels and 16 grouped labels. Only 621 of the
150,000 training labels change, including 126 validation rows.

The same mapping is applied to train and test; category vocabularies come from
fitting rows only. This is the controlled direct-category ablation: historical
rent feature values and historical grouping keys retain their original form.
It does not rebuild model-specific histories under the merged labels. No row
is removed and rent targets are unchanged. Mapping source:
`workflows/tabpfn/analysis/flat_model_groups.py`.

The grouped ensemble has the better observed development RMSE, by only
0.1813 SGD. This repeatedly examined holdout does not establish a Kaggle gain.
Keep the original-category submission as a comparison. Neither new file has
an observed Kaggle score, and the runner does not upload either file.

## Files to submit

The ready CSVs are tracked under `reports/submissions/tabpfn_ensemble_20261106/`.
The original runtime's generated outputs were under `output/tabpfn-ensemble/`;
that ignored directory is not guaranteed to exist in another checkout:

| File | Purpose |
|---|---|
| `submission_ensemble_grouped_150k.csv` | Recommended grouped-category candidate |
| `submission_ensemble_original_150k.csv` | Regenerated original-category baseline |
| `validation_scores.csv` | Fresh component and ensemble validation scores |
| `audit.json` | Submission schema, IDs, numerical comparisons and hashes |
| `kaggle_submissions.zip` | Both submissions with metrics and provenance |

Both CSVs contain exactly `Id,Predicted`, with 50,000 unique IDs in the sample
submission's order. Predictions are finite and positive. Each newly fitted
tree checkpoint is reloaded and its predictions checked before blending.

## Reproduce on main

Use the existing CUDA environment and the checkpoint described in the
[standalone TabPFN guide](tabpfn_best.md). Prepare its isolated runtime first;
the ensemble runner uses main's tree modules and that runtime's TabPFN modules
in separate processes:

```powershell
python run_tabpfn_best.py prepare --assets-from .
python run_tabpfn_ensemble.py all --output output/tabpfn-ensemble-fresh
```

The `all` stage refits both category variants on validation and full training
data. CPU tree training and GPU TabPFN inference run with fixed settings.
The original TabPFN component in this audit was imported from the separately
verified standalone run; its full-data predictions were freshly computed in
this audit. `all` independently refits that component as well.

Individual stages are `trees-validate`, `trees-predict`, `tabpfn-validate`,
`tabpfn-predict` and `export`. Add `--variants grouped` to fit/export only the
grouped model, and `--runtime output/another-runtime` to choose a prepared
TabPFN runtime. Existing model predictions are reused only when their recorded
identity and hashes agree. Use a fresh output directory after changing code,
inputs or the numerical environment.

The tracked evidence is in `review/checks/tabpfn-ensemble/`. Raw datasets,
historical reports and earlier submissions are preserved.

The evidence retains the exact training runner snapshot and component hashes.
The current runner additionally creates a missing full-feature report directory;
this setup change does not change feature values, fitting or predictions.

Final CSV copies are committed on main: [grouped candidate](../../reports/submissions/tabpfn_ensemble_20261106/submission_ensemble_grouped_150k.csv), [original baseline](../../reports/submissions/tabpfn_ensemble_20261106/submission_ensemble_original_150k.csv), and [publication record](../../reports/submissions/tabpfn_ensemble_20261106/publication.json). These copies have the same hashes as the audited runtime outputs.
