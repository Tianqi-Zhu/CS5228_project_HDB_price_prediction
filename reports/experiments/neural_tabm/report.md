# TabM-style neural rental model

Evaluation uses the same expanding fixed-origin folds as the spatiotemporal experiment. Historical target features are frozen at each origin.

The lowest pooled RMSE is **S$542.78** from **blend_selected_oof**. The historical tree scores S$543.10.

The final blend weight selected from pooled out-of-fold predictions assigns **0.94** to the tree and **0.06** to the neural model. Because the same out-of-fold predictions select and report this weight, `blend_selected_oof` is a model-selection estimate rather than an unbiased performance estimate.

```text
           variant     n  rmse_sgd  mae_sgd  bias_sgd
blend_selected_oof 79817    542.78   414.75    -14.60
   historical_tree 79817    543.10   414.49     -7.02
          blend_50 79817    567.63   439.15    -70.16
              tabm 79817    646.37   506.07   -133.30
```

## Fold scores

```text
validation_start            variant     n  rmse_sgd  mae_sgd  bias_sgd
         2023-01    historical_tree 17754    639.02   524.72    -72.01
         2023-07    historical_tree 26905    563.35   437.04      6.92
         2024-04    historical_tree 35158    469.61   341.57     15.13
         2023-01               tabm 17754    788.60   646.05   -367.72
         2023-07               tabm 26905    645.58   496.57     89.66
         2024-04               tabm 35158    561.71   442.66   -185.55
         2023-01           blend_50 17754    682.55   564.36   -219.87
         2023-07           blend_50 26905    579.18   445.01     48.29
         2024-04           blend_50 35158    489.41   371.43    -85.21
         2023-01 blend_selected_oof 17754    640.35   526.91    -89.75
         2023-07 blend_selected_oof 26905    562.31   435.62     11.88
         2024-04 blend_selected_oof 35158    468.83   342.14      3.09
```

The neural model uses embeddings for property/location categories and a shared MLP with per-member rank-one adapters. Member predictions are averaged at inference.