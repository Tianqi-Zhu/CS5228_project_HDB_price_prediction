from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class PreprocessorConfig:
    """Configuration for HDB rent preprocessing."""
    target_col: str = "monthly_rent"
    date_col: Optional[str] = None


class HDBPreprocessor:
    """
    Handles:
    - data loading
    - basic EDA summaries
    - data cleaning
    - feature engineering
    - sklearn preprocessing pipeline construction
    """

    def __init__(self, config: PreprocessorConfig | None = None):
        self.config = config or PreprocessorConfig()

    def load_data(self, path: str | Path) -> pd.DataFrame:
        """Load raw HDB rental data."""
        path = Path(path)

        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        if path.suffix.lower() in {".xlsx", ".xls"}:
            return pd.read_excel(path)

        raise ValueError(f"Unsupported file type: {path.suffix}")

    def basic_eda(self, df: pd.DataFrame) -> dict:
        """Return lightweight EDA statistics."""
        return {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isna().sum().to_dict(),
            "duplicates": int(df.duplicated().sum()),
            "numeric_summary": df.describe(include="number").to_dict(),
        }

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply basic cleaning rules."""
        df = df.copy()

        # Remove exact duplicate rows.
        df = df.drop_duplicates()

        # Standardize column names.
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
            .str.replace(r"[^\w]+", "_", regex=True)
        )

        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add HDB-specific derived features.

        Examples you may add later:
        - remaining_lease
        - lease_age
        - rent_year / rent_month
        - distance_to_mrt
        - distance_to_cbd
        - floor_area_per_room
        """
        df = df.copy()

        if self.config.date_col and self.config.date_col in df.columns:
            dt = pd.to_datetime(df[self.config.date_col], errors="coerce")
            df["rent_year"] = dt.dt.year
            df["rent_month"] = dt.dt.month

        return df

    def split_features_target(self, df: pd.DataFrame):
        """Split dataframe into X and y."""
        target = self.config.target_col

        if target not in df.columns:
            raise KeyError(f"Target column '{target}' not found.")

        X = df.drop(columns=[target])
        y = df[target]
        return X, y

    def infer_feature_types(self, X: pd.DataFrame):
        """Infer numeric and categorical columns."""
        numeric_cols = X.select_dtypes(include="number").columns.tolist()
        categorical_cols = X.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        return numeric_cols, categorical_cols

    def build_transformer(self, X: pd.DataFrame) -> ColumnTransformer:
        """Build sklearn preprocessing transformer."""
        numeric_cols, categorical_cols = self.infer_feature_types(X)

        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, numeric_cols),
                ("categorical", categorical_pipeline, categorical_cols),
            ],
            remainder="drop",
        )

    def prepare(self, df: pd.DataFrame):
        """Full preprocessing flow before train/test split."""
        df = self.clean(df)
        df = self.engineer_features(df)
        X, y = self.split_features_target(df)
        transformer = self.build_transformer(X)

        return X, y, transformer
