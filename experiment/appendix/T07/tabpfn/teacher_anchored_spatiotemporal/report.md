# TabPFN-anchored spatial–temporal correction

## Decision

The hybrid **does not improve** the existing TabPFN ABCDE equal forecast. Its
historical validation RMSE improved by S$5.891, but the locked correction raised
June 2024–March 2025 development RMSE by S$3.508–4.389 across all three seeds.
It worsened all ten months for every seed. No submission or existing model was
changed.

| Forecast | June 2024–March 2025 RMSE (S$) | Change vs TabPFN (S$) |
| --- | ---: | ---: |
| TabPFN ABCDE equal | 454.704 | — |
| Hybrid seed 42 | 459.093 | +4.389 |
| Hybrid seed 43 | 458.768 | +4.064 |
| Hybrid seed 44 | 458.212 | +3.508 |

## Experiment

The teacher is the existing A/B/C/D/E equal forecast at three fixed origins.
The two historical origins each have 2,000 sampled, population-weighted
out-of-origin forecasts. Their TabPFN contexts end before the query origins;
the historical A adapters were also fitted without query labels. The
transformer feature bundle supplies aligned, origin-causal 12-month rental
sequences. The teacher prediction is always the anchor and is never recomputed
from a context containing the query label.

The new correction has 9,660 parameters. A property query formed from
coordinates, physical attributes, local-history support, the TabPFN deviation
from the common temporal baseline, flat type, and forecast context attends to
12 monthly rental-history tokens. A learned sigmoid gate multiplies a capped,
zero-initialized output. Training minimizes weighted squared teacher error plus
a squared-correction penalty. The experiment tests penalties 0.25, 1, and 4
and nine fixed checkpoints through epoch 35. The September 2022 origin fits
the correction; the July 2023 origin selects penalty and epoch. The selected
configuration, penalty **0.25** and epoch **35**, is refitted on both historical
origins for three seeds. The June 2024 origin supplies only a retrospective
evaluation. GPU training started after two capacity checks 15 seconds apart,
each requiring at least 8 GiB free and at most 35% utilization.

| Historical origin | Teacher RMSE (S$) | Teacher bias (S$) | Role |
| --- | ---: | ---: | --- |
| September 2022 | 633.539 | +45.473 | Correction fit |
| July 2023 | 567.802 | +116.412 | Selection; selected hybrid 561.911 |
| June 2024 | 454.704 | +0.988 | Retrospective evaluation |

## What failed to transfer

The earlier teacher increasingly overpredicted rent late in the July 2023
forecast window. The correction learned to subtract rent. At the June 2024
origin the teacher's aggregate bias was near zero, while the hybrid's mean
correction ranged from **−S$46.63 to −S$53.97** across seeds. The mean monthly
correction grew more negative with forecast horizon. This is consistent with a
period-specific bias adjustment transferring poorly; it does not prove that
the spatial attention mechanism itself is harmful.

A posthoc *diagnostic* subtracting the correction's within-month mean still
raised outer RMSE to S$455.88–455.99. Thus removing its national monthly shift
did not reveal a useful spatial correction in this period. This diagnostic was
examined after seeing the outer result and is **not** an independently selected
candidate. A direct scratch-T0 blend is also unstable across origins: the
weighted optimal nonnegative blend coefficient from September 2022 is zero,
from July 2023 about 0.594, and from June 2024 zero. These are descriptive
within-origin calculations, not deployable weights.

The dataset provides only 4,000 historical teacher forecasts at two origins,
and the later development window has been repeatedly inspected. Those limits
make small positive validation gains unreliable. A future teacher-prior study
would need more chronologically generated teacher forecasts across several
distinct market regimes and a newly held-out period. Frozen TabPFN embeddings
and reverse stacking remain separate, untested variants; this negative result
does not rule them out.

## Reproduction and artifacts

Run `.venv/Scripts/python.exe -u analysis/run_teacher_anchored_spatiotemporal.py --device cuda`.
The run records [settings](settings.json), [historical selection](selection.csv),
[locked choice](selection.json), [seed scores](scores.csv), [monthly scores](monthly_scores.csv),
[origin and source checks](verification.json), and row predictions for each
seed. The code and artifacts are isolated from all existing submissions.
