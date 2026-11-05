# Conservative low-rent candidate filter

The HDB source series is [Renting Out of Flats from Jan 2021](https://data.gov.sg/datasets/d_c9f57187485a850908655db0e8cfe651/view). HDB handles [renting out bedrooms](https://www.hdb.gov.sg/managing-my-home/home-ownership/renting-out-a-flat-or-bedrooms/renting-out-bedrooms/application-process) as a distinct approval process. The local rental rows contain no field for number of bedrooms rented. These files therefore identify **unusually low reported rents**, not verified bedroom-only leases.

The conservative flag applies to 3-room, 4-room, 5-room, and executive records. A row is flagged only when its monthly rent is strictly below 50% of both:

1. The median rent of at least 30 records in the same town, normalized flat type, and calendar quarter.
2. The median rent of at least five records in the same town, street, block, normalized flat type, flat model, floor area, and calendar year.

Medians include the row under review. With at least five close comparables and a strict half-median cutoff, this is a conservative screen; it still cannot prove the reason for a low declared rent. The cutoff is an analysis choice, not an HDB rule. It should be fitted again inside each training fold before a validation experiment, leaving the validation fold untouched.

Outputs from `data/train.csv`:

- `train_without_low_rent_candidates.csv`: 149,695 rows, same schema and order as the source, after 305 candidate rows are removed.
- `low_rent_candidates.csv`: 305 removed rows with source line numbers, peer counts, medians, and ratios for review.
- `broad_low_rent_review.csv`: 921 rows below half the town/type/quarter median, including those without sufficient close block comparables.
- `all_row_flags.csv`: one line per source row with both flags.
- `summary.json`: counts and rule parameters.

The 305 conservative candidates are 3 in 2021, 23 in 2022, 128 in 2023, 148 in 2024, and 3 in January-March 2025. There are no flagged executive rows under this rule. Of the 305, 303 also appear in the earlier `rent_per_room_500/removed_below_500.csv` output. The original `data/train.csv` remains the reference dataset.

Reproduce with `.venv/Scripts/python.exe analysis/flag_possible_partial_rentals.py`.
