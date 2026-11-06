[Chinese version](README.zh-CN.md)

# F1 final-method reproduction

| Item | Value |
|---|---|
| Provenance | results/hdb-rental-final-ensemble @ 6f97e107ed2db72347c3d7d1f85b63999b3fcbbb |
| Scope | Structural blend F1 only; historical/macro variants are on the appendix branch |
| Standard-library verification | `python reproduction/f1/verify.py` |
| CPU prediction reconstruction | `python reproduction/f1/reproduce.py` (requires NumPy) |
| Full retraining | `python reproduction/f1/train.py` (CUDA and full training environment; prior full-training acceptance is in the running guide) |
| Reload new checkpoints | `python reproduction/f1/check_trained.py` |
| Count | 12 prediction streams, 15 base predictors, one CSV |
| Data | Uses this branch's root data/ directly; no second copy |
| Code isolation | runtime/ retains original flat modules; config only adapts the project root; do not mix with root grouped modules |
| Known limitations | Original scores are user-screenshot values with public/private split unknown; reconstruction is not fresh training; new retraining CSVs have no known Kaggle score |

## Fixed-recipe Y/J validation generation chain

`validate.py` independently rebuilds features from this branch's `data/`, using pre-origin blend weights fixed in [validation_recipe.json](validation_recipe.json) and rounds in [training_recipes.json](training_recipes.json), training 15 base models per window. Query labels are zeroed in all fitting calls; they do not stop training. No new search or weight fitting occurs. Each saved model is immediately reloaded and predictions compared.

```powershell
python reproduction/f1/validate.py --window Y --output-dir output/f1-validation/new-Y
python reproduction/f1/validate.py --window J --output-dir output/f1-validation/new-J
python -m analysis.main validation-refits
```

The first two commands require CUDA/training dependencies and a nonexistent output directory. The third uses only CPU to verify saved component predictions, code/data hashes, exact generating commits and blends, without fresh training or large checkpoints.

| Archive | Content |
|---|---|
| [Y](validation_evidence/Y/validation_run.json), [J](validation_evidence/J/validation_run.json) | Generating commits, 30 training/reload records, input columns, fixed rounds and data/code hashes |
| [Summary verification](validation_evidence/audit.json) | Recomputed IDs/labels, raw data, commit code and generated CSVs for both windows |
| Per-window `components.npz` | 12 logical streams and support groups for offline reconstruction, avoiding duplicate full-model/feature-cache commits |

Large new checkpoints remain in local `output/five_item_validation/`; the first two commands reproduce training. New chains and old prediction archives are identified separately, without claiming recovery of old caches' original generating commits. Y/J remain inspected development sets; see [scenario boundaries](../../docs/methods/prediction-scenario.md).
