[Chinese version](FEATURE_ABLATION.zh-CN.md)

# Single-field/group deletion and refitting ablations

This round measures actual RMSE increments with the default27 inputs/fixed LightGBM.
Full baseline, 27 single, eight group and three joint deletions: 39 configurations×four windows=156 fits.
Match fitting/validation rows per fold; pool three disjoint annual windows and list overlapping16-month stress separately.

```text
Delta RMSE = deleted-field refit RMSE − full27-input RMSE
```

Positive means worse after deletion; retained correlated features/derived histories can replace information,
so contributions are not additive, causal or complete source removal.

| File | Content |
| --- | --- |
| `feature_effects.csv` | Three-window, latest/stress changes and stability for every deletion |
| `fold_effects.csv` | Settings, counts, RMSE/MAE/bias and paired differences for 156 fits |
| `monthly_paired_sse.csv` | Monthly counts and baseline/candidate SSE for resampling |
| `error_diagnostics.csv` | Diagnostics by month, horizon, town, type, history support/staleness and true-rent band |
| `group_deletion.png` | Group summary plot; large time+all-history joint deletion listed separately in title |
| `results.json` | Complete fields, settings, windows, source hashes, versions and limits |

Intervals use 2000 paired circular three-month contiguous-block draws of fixed predictions, describing month sensitivity only.
They omit fitting randomness, future-market uncertainty and multiple comparisons. Labels were previously inspected,
so results remain exploratory interpretation/selection, not independent testing. Rankings do not automatically change the default.

```powershell
python -m analysis.main feature-evidence
```

Full narrative:[current project report](BASELINE_27_FEATURES.md).
