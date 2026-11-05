# Hierarchical spatiotemporal evaluation

The experiment uses expanding fixed-origin validation. No validation target is used as a lag or target encoding, so each fold simulates a batch multi-month forecast.

## Pooled result

The selected variant is **historical**, with pooled RMSE **S$543.10** versus **S$549.10** for the static spatial baseline (change -6.01).

```text
                  variant     n  rmse_sgd  mae_sgd  bias_sgd  delta_rmse_vs_spatial_baseline_sgd
               historical 79817    543.10   414.49     -7.02                               -6.01
historical_spatiotemporal 79817    543.13   413.63      1.66                               -5.97
   spatial_spatiotemporal 79817    548.83   425.74    -50.47                               -0.27
         spatial_baseline 79817    549.10   426.36    -53.02                                0.00
```

## Fold results

```text
validation_start validation_end_exclusive                   variant  n_fit  n_validation  horizon_months  rmse_sgd  mae_sgd  bias_sgd
         2023-01                  2023-07          spatial_baseline  70183         17754               6    642.74   531.39    -86.61
         2023-01                  2023-07    spatial_spatiotemporal  70183         17754               6    642.30   530.73    -84.82
         2023-01                  2023-07                historical  70183         17754               6    639.02   524.72    -72.01
         2023-01                  2023-07 historical_spatiotemporal  70183         17754               6    638.08   523.76    -64.87
         2023-07                  2024-04          spatial_baseline  87937         26905               9    567.69   448.53    -39.82
         2023-07                  2024-04    spatial_spatiotemporal  87937         26905               9    567.66   448.23    -37.82
         2023-07                  2024-04                historical  87937         26905               9    563.35   437.04      6.92
         2023-07                  2024-04 historical_spatiotemporal  87937         26905               9    563.85   436.14     16.38
         2024-04                  2025-04          spatial_baseline 114842         35158              12    478.82   356.36    -46.16
         2024-04                  2025-04    spatial_spatiotemporal 114842         35158              12    478.44   355.51    -42.81
         2024-04                  2025-04                historical 114842         35158              12    469.61   341.57     15.13
         2024-04                  2025-04 historical_spatiotemporal 114842         35158              12    469.90   340.79     23.99
```

Historical features are prior-month-only expanding summaries during fitting and are frozen at each forecast origin. The spatiotemporal correction recursively forecasts planning-area × flat-type residuals from group lags, flat-type and global lags, seasonality, linear time, and inverse-distance neighboring-area lags.