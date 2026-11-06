# First-pass S$1,500 monthly rent filter

Starting from `data/train.csv`, normalize the spelling of `FLAT_TYPE` (`3 room` and `3-room` are the same). Remove rows with `MONTHLY_RENT <= 1500` for 3-room, 4-room, 5-room, and executive flats. This includes rents exactly equal to S$1,500. Leave 1-room and 2-room flats untouched.

- `train_filtered.csv`: 146,232 retained rows, in original order and with the original training schema.
- `removed_monthly_rent_le_1500.csv`: 3,768 removed rows, with original CSV line number and filter reason.
- `summary.json`: counts by approval year and flat type.

This is an experimental low-rent screen. The source HDB series is [Renting Out of Flats from Jan 2021](https://data.gov.sg/datasets/d_c9f57187485a850908655db0e8cfe651/view); HDB's [renting out bedrooms process](https://www.hdb.gov.sg/managing-my-home/home-ownership/renting-out-a-flat-or-bedrooms/renting-out-bedrooms/application-process) is separate. The input rows have no field stating how many bedrooms were rented, so a removed row is **not** a confirmed bedroom-only lease. In 2021, 1,737 rows meet this price rule; some could be valid lower-cost whole-flat rentals.

Reproduce: `.venv/Scripts/python.exe analysis/filter_whole_flat_rent_1500.py`.
