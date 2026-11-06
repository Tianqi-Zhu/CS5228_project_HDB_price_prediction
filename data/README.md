[Chinese version](README.zh-CN.md)

# Competition data

This directory preserves the original competition CSVs. Counts below refer to data rows in the current local version, excluding headers.
The prediction target is monthly rent `MONTHLY_RENT`, in Singapore dollars (SGD).

| File | Rows | Purpose |
| --- | ---: | --- |
| [train.csv](train.csv) | 150,000 | Labelled rental samples; split chronologically into fitting, selection and local-validation windows |
| [test.csv](test.csv) | 50,000 | Kaggle query inputs; excludes `MONTHLY_RENT` |
| [example-submission.csv](example-submission.csv) | 50,000 | Output template with `Id,Predicted` columns; preserve its IDs and row order |
| [auxiliary/sg-hdb-block.csv](auxiliary/sg-hdb-block.csv) | 9,464 | Building coordinates, completion year, floors and geographic areas |
| [auxiliary/sg-mrt-stations.csv](auxiliary/sg-mrt-stations.csv) | 243 | MRT locations and status for transport-accessibility features |
| [auxiliary/sg-schools.csv](auxiliary/sg-schools.csv) | 337 | School types and locations for nearby-facility features |
| [auxiliary/sg-shopping-malls.csv](auxiliary/sg-shopping-malls.csv) | 113 | Mall locations for distances and other spatial features |
| [auxiliary/sg-coe-prices.csv](auxiliary/sg-coe-prices.csv) | 539 | Supplied macro information including month, vehicle category and COE premiums |
| [auxiliary/sg-stock-prices.csv](auxiliary/sg-stock-prices.csv) | 91,642 | Supplied macro information including date, security identifier and prices |

`train.csv` has one additional column, `MONTHLY_RENT`, compared with `test.csv`; other raw fields cover date, town, block, street, flat type,
flat model, floor area, furnishing, lease commencement year and fees. Each round's configuration determines which fields are used.
See [feature definitions](../docs/methods/baseline_features.md) for cleaning and baseline features,
and [final method](../experiment/main/M06/README.md) for the final approach.

## Usage boundaries

- Local validation uses historical portions of `train.csv`; rental statistics respect each experiment's information cutoff.
- `test.csv` supplies final queries and submission inputs, not rent labels for training or selection.
- Report historical-only methods separately from methods using supplied query-month macro observations. A future-month table is not evidence that its values were observable in real time.
- The final three submissions do not depend on independent-source audit data in `.cache/public_rents/` or public-rent matching results.

Write cleaned data, derived features, caches and predictions to the relevant artifact directories; do not overwrite these raw CSVs.
Existing `source_sha256` records and submission-component manifests identify input versions. Recheck after replacing CSVs; old metrics cannot simply be reused.
See the [running guide](../docs/guides/running_experiments.md) for environments and commands.

## Byte-level provenance

The shipped CSV bytes preserve the Windows CRLF inputs recorded by the frozen training recipes. Original source Git blobs used LF through Git conversion; after CRLF-to-LF normalization all nine files are identical to source commit `048eca86`. The current `.gitattributes` disables conversion so recorded recipe SHA256 hashes survive clean checkouts. No rows or values were changed.
