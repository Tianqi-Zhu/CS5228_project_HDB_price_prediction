# Historical-summary concat Transformer

EDA and fixed-model feature engineering motivate the historical/market-relative inputs. This experiment asks whether an independently implemented feature Transformer benefits from greater width/depth on those same inputs. It is a standalone architecture exploration; F1 and F2 remain separate pipelines.

## Inputs and target

Use the final PLR-FT member's **adjusted_history_no36_62**, not the differently composed62column feature-ladder schema. Raw input groups:19property/time/location/amenity,8block/area×type history,20market-relative history,10anniversary12/24month±1month fields,5cohort24-adjusted anchor fields. The15anniversary12/24/36 fields in the ladder become10after dropping36month fields, while5anchor fields are added. Exact names: [feature_list.txt](feature_list.txt).

Fit-only encoder: categorical vocab/unknown0, count log transform where defined, numeric medians/quantiles and missing flags. Development bank has95numeric+4categorical inputs after encoding. Final vocabulary refits on150000 and can gain categories. Static facility fields follow the team's supplied-snapshot policy. Training historical labels are strictly earlier than each row's month; validation histories freeze before2024-04. No query labels update histories.

The anchor is the cohort24relative rent ratio shrunk toward local history, multiplied by past3month flat-type market level. Target=(actual rent−anchor)/1000. Prediction=1000×predicted residual+anchor, exactly once.

## Architecture

Every numerical field has a learned vector-valued linear token; each category has an embedding. Two separate two-layerMLPs map predefined subsets of encoded historical inputs to relative-history and anniversary/anchor summary tokens. Their membership is explicit in model.py and may overlap: these are learned views, not disjoint feature groups. Append these2tokens to the feature/CLS sequence;3or6pre-norm Transformer blocks predict the signed residual fromCLS. They add no information beyond the same feature set. Final width128,6blocks,4heads,FFmultiplier2,dropout.05,916225parameters.

Earlier DBS-LSTM/COE-GRU summary tokens were another exploration and are intended for appendixX05; they are not inputs to this final model. PLR numerical embeddings are not used here.

## Capacity and training selection

| Width | Layers | Parameters | Best epoch | Total epochs | Annual validation RMSE |
|---:|---:|---:|---:|---:|---:|
|48|3|90865|15|20|459.9921|
|128|3|518785|8|13|459.8288|
|128|6|916225|8|13|459.2189|

All use114842fit<=2024-03,35158validation2024-04..2025-03,AdamW3e-4,weight_decay1e-4,batch512,gradientclip1,seed42,warmup1epoch,cosine40epochs,min/startLRratio.01,max40,earlystopping5. Best saved weights are reloaded and all validation predictions replayed. Full curves/settings/environments are in each model's selection.json/protocol.json; table in capacity_comparison.csv. Width/depth were compared with other settings fixed; this is a limited configuration search. The capacity gain is small and single-run; no claim of general architectural superiority.

The outer period has been repeatedly inspected for development and bestepoch selection. Teammate PLR's462.07 uses an earlier-selected fixedepoch recipe and different architecture/target standardization; same inputs do not establish an equal-budget architecture ranking. The team's451.03 is an ensemble and must not be labeled a singleTransformer control.

## Final all-label refit and submission

Choose128×6and8epochs from validation. From fresh weights, refit preprocessing on150000labels and train exactly8epochs, no validation or earlystopping; keep the original cosine40schedule rather than compressing decay into8epochs. Generate50000test predictions in sample-ID order. TestMarch2025 excludesMarchlabels from its history; later queries freeze history throughMarch2025.

Previous local validationRMSE459.2189 and submitted **Public Score418.38657** are different metrics on different populations, not an estimated constant offset. Public score is transcribed from the user's Kaggle screenshot, not independently queried; private score unknown. The archived CSV checksum links the score record to the file the user identified. Final test labels are not available locally.

[Reproduction entry](../../../../reproduction/concat_transformer/README.md) includes the original final weights, preprocessing, frozen prepared bank, configuration, CSV and checksums. Rebuild uses shared main feature code. Model selection and full-refit files are separate. No actual historical hyperparameter search is rerun by integrity/replay commands.
