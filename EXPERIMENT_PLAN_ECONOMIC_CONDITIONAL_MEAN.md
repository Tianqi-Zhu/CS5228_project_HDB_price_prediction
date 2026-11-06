# Joint temporal economic conditioning of expected rent

Test the conditional mean of individual monthly rent, not the observed mean of
the future query month. Reuse the original causal small-transformer folds.

Prediction is the unchanged causal local rent baseline plus the jointly trained
property-transformer residual, a temporal conditional-mean shift, and an
eight-dimensional interaction between temporal state and property state.
A small eight-unit LSTM encodes twelve observed monthly histories. Its hidden
state and known forecast horizon/calendar feed the mean parameters. Initialize
the additional output exactly at zero. Every neural parameter remains trainable.

Matched designs: rent-only histories versus rent+COE histories. The latter adds
COE A/C log premiums and changes; stocks are masked in both designs. Preserve
the original extra month of macro publication buffer and startup missingness.
No target-month macro data, query rents, or separately generated LSTM forecasts
enter an input. No new mean based on query labels is constructed.

Use the unchanged FT property backbone (128 dimensions, two blocks), AdamW
0.0003, weight decay0.001, dropout0.1, batch256, gradient clip1, BF16 CUDA,
max30 epochs, patience5, seed42 for earlier selection. The individual-rental
MSE adds0.01 times the squared temporal correction. Preserve the exact same
initial property parameters and shuffle stream for the two designs.

September2022-June2023 and July2023-April2024 are the earlier two2000-query,
population-weighted windows. Select each design's fixed refit epoch from the
rounded median of its two earlier best epochs. Proceed to the already reused
June2024-March2025 development period only if rent+COE beats both the matched
rent-only design and existing T0 on each earlier window. A failure completes
this bounded experiment without a later-period weight/architecture search.
If it passes, refit both designs at seeds42/43/44 and compare with T0 and the
locked rank23 blend; report monthly bias, conditional shifts, and paired
three-month block intervals. No later labels select training settings.

Preserve original code, fold bundles, checkpoints, and submission CSVs.
Use a separate report/cache directory, source hashes, resumable checkpoints,
prediction replay, CPU tests and smoke, CUDA smoke, and the inherited two-sample
GPU gate (8GiB free, utilization<=35%, samples15 seconds apart).
Only51 labeled months exist; all later periods are development evidence.
