# HDB Rent Prediction

Starter codebase for an HDB monthly rent prediction project.

## Structure

```text
hdb_rent_prediction/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── auxiliary/
│       ├── xxx.csv
│       └── ...
├── models/
├── reports/
│   └── figures/
├── src/
│   ├── __init__.py
│   ├── preprocessor.py
│   ├── visualizer.py
│   ├── trainer.py
│   ├── evaluator.py
│   └── main.py
├── requirements.txt
└── README.md
```

## Data layout

### `data/train.csv`
Main labeled training dataset.

Expected to contain the prediction target, currently configured as:

```text
monthly_rent
```

### `data/test.csv`
External / competition test set.

This file normally contains the same input features as `train.csv` but does
not need to contain `monthly_rent`.

### `data/auxiliary/`
Additional datasets used for feature engineering.

Examples:

```text
data/auxiliary/mrt.csv
data/auxiliary/schools.csv
data/auxiliary/malls.csv
data/auxiliary/hdb_blocks.csv
```

These datasets can later be merged into the train and test datasets using
location, postal code, block/street information, or other suitable keys.

## Modules

### `preprocessor.py`

Responsible for:

- loading CSV / Excel data
- lightweight EDA
- cleaning
- feature engineering
- feature / target separation
- numeric and categorical preprocessing

### `visualizer.py`

Responsible for:

- target distributions
- feature-vs-rent plots
- grouped rent plots
- reusable EDA visualizations

### `trainer.py`

Responsible for:

- internal train/validation split
- sklearn model pipelines
- baseline regression model
- training
- model persistence

### `evaluator.py`

Responsible for:

- MAE
- RMSE
- R²
- residual inspection
- actual-vs-predicted output

### `main.py`

Runs:

```text
data/train.csv
        ↓
cleaning / feature engineering
        ↓
EDA
        ↓
internal train-validation split
        ↓
model training
        ↓
validation evaluation
        ↓
optional prediction on data/test.csv
```

Auxiliary CSVs can be incorporated during feature engineering once their
schemas and join keys are known.

## Running

Install dependencies:

```bash
pip install -r requirements.txt
```

Then run from the project root:

```bash
python src/main.py
```

## Suggested HDB Features

Potential main or auxiliary features include:

- town
- block
- street_name
- flat_type
- floor_area_sqm
- storey_range
- latitude
- longitude
- lease_commence_date
- remaining_lease
- distance_to_nearest_mrt
- distance_to_cbd
- nearby_school_count
- nearby_mall_count
- rental_year
- rental_month

For chronological rental records, a time-based validation split may be more
appropriate than a random train/validation split.
