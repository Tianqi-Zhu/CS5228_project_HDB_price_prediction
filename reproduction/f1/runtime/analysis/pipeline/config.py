"""Project paths and the fixed settings used by the midterm experiments."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
DATA = ROOT / "data"
AUX = DATA / "auxiliary"
REPORTS = ROOT / "reports"
EDA_REPORTS = REPORTS / "eda"
STUDY_REPORTS = REPORTS / "studies"
EXPERIMENT_REPORTS = REPORTS / "experiments"
OUTPUT = ROOT / "output"
CACHE = ROOT / ".cache"
MODELS = OUTPUT / "models"
SUBMISSIONS = OUTPUT / "submissions"
TARGET = "MONTHLY_RENT"
DATE = "RENT_APPROVAL_DATE"
SOURCES = [DATA / "train.csv", DATA / "test.csv", DATA / "example-submission.csv"]
VALIDATION_START = "2024-04"

FEATURE_SETS = {
    "A_property": ["FLAT_TYPE_CLEAN", "FLAT_MODEL_CLEAN", "FLOOR_AREA_SQM"],
    "B_property_time": [
        "FLAT_TYPE_CLEAN", "FLAT_MODEL_CLEAN", "FLOOR_AREA_SQM",
        "APPROVAL_MONTH_INDEX", "APPROVAL_MONTH_OF_YEAR", "FLAT_AGE_YEARS",
    ],
    "C_property_time_town": [
        "FLAT_TYPE_CLEAN", "FLAT_MODEL_CLEAN", "FLOOR_AREA_SQM",
        "APPROVAL_MONTH_INDEX", "APPROVAL_MONTH_OF_YEAR", "FLAT_AGE_YEARS", "TOWN_CLEAN",
    ],
}
CATEGORICAL = ["FLAT_TYPE_CLEAN", "FLAT_MODEL_CLEAN", "TOWN_CLEAN"]
NUMERIC = ["FLOOR_AREA_SQM", "APPROVAL_MONTH_INDEX", "APPROVAL_MONTH_OF_YEAR", "FLAT_AGE_YEARS"]
MODEL_FAMILIES = ("ridge", "random_forest", "catboost", "lightgbm")

CATBOOST_SETTINGS = {
    "iterations": 400,
    "depth": 6,
    "learning_rate": 0.05,
    "l2_leaf_reg": 3.0,
    "loss_function": "RMSE",
    "random_seed": 42,
    "thread_count": 4,
    "verbose": False,
    "allow_writing_files": False,
}
FAMILY_SETTINGS = {
    "ridge": {"alpha": 10.0},
    "random_forest": {
        "n_estimators": 160,
        "max_depth": 18,
        "min_samples_leaf": 5,
        "max_features": 0.8,
        "random_state": 42,
        "n_jobs": 4,
    },
    "lightgbm": {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 30,
        "reg_lambda": 3.0,
        "random_state": 42,
        "n_jobs": 4,
        "verbosity": -1,
    },
}
