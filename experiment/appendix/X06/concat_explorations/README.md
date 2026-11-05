# Earlier Transformer, cross-attention and TabFM exploration

This is the development history preceding the final same-feature concat Transformer. It complements the other contributors’ existing X06 evidence.

## Historical macro-token ablation

`feature_transformer_v4/runs/v4-macro-primary-s42/summary.json` records historical validation on 35,158 rows: no macro tokens 468.9730; DBS-LSTM plus COE-GRU tokens 467.4027. Both selected epoch 23 using the development selection procedure with a 60-epoch cosine schedule. These are historical results under an earlier feature/training protocol, not the final 62-column recipe.

## Matched architecture comparison

The fresh CPU comparison uses the same 88,192 fit rows, 35,158 historical validation rows, seed, fresh common weights/RNG state, fixed 23 epochs and 3,979 updates. Read `cross_attention_matched467_v12/REPORT.md` and registration/provenance for exact controls.

| Architecture | Historical validation RMSE |
|---|---:|
| Concat baseline | 467.6968 |
| Cross-attention to summary tokens | 469.4115 |
| Cross-attention to sequence tokens | 468.6702 |

Neither cross-attention arm improved the fresh concat control in this experiment. The older MPS score 467.4027 is a reference, not the fresh matched control. This evidence supported continuing with concat; it does not establish general superiority across architectures or datasets.

`cross_attention_v11` is an earlier separate protocol; do not mix its scores with the matched comparison. Its archived settings explain why apparently conflicting historical results exist.

## Later training and foundation-model attempts

- `concat_raw_v14`, `concat_warmup_v15`, `concat_depth_v16`: available training/selection records before aligning with the team’s final feature recipe.
- `tabfm_v10`: pretrained TabFM context inference and small screens. Its README distinguishes the 2023–24 development period, the later historical benchmark and smoke-only queries.
- `tabfm_finetune_v13`: available local LoRA/head fine-tuning records and curves. A curve or a directory alone does not establish a completed final search. The separately reported SoC v16 log excerpts were not downloaded as a complete evidence bundle here; no final cluster score is inferred from them. The original v16 pipeline source is retained for inspection.

The final main/M05 model uses 62 raw features from the team’s adjusted-history pipeline, two MLP historical summary tokens, width 128 and six layers. Its 459.2189 validation RMSE and user-reported Kaggle public 418.38657 are documented in the separate main integration; this appendix does not claim that a single isolated architecture change caused that difference.

[Source snapshot and verification](../../../../reproduction/concat_explorations/README.md). These are archived records; no model training or queue submission occurred during integration.
