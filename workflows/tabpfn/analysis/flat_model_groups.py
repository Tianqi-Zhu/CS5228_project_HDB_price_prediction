"""User-proposed HDB model grouping; this is not an official HDB taxonomy."""
import pandas as pd

MERGES = {
    "type s1": "type s", "type s2": "type s",
    "model a maisonette": "maisonette", "premium maisonette": "maisonette",
    "improved maisonette": "maisonette",
}
VARIANTS = ("grouped", "original_plus_group")


def normalized_model(values):
    return values.astype("string").str.strip().str.lower().str.replace(r"\s+", " ", regex=True)


def model_group(values):
    return normalized_model(values).replace(MERGES)


def apply_model_group(x, q, variant, train_models=None, query_models=None):
    """Change only the direct model category; preserve every other feature."""
    if variant not in VARIANTS:
        raise ValueError(variant)
    x, q = x.copy(), q.copy()
    # Raw labels retain subtypes unseen in training, before categorical encoding
    # turns those subtypes into missing values. Group vocabularies still use fit only.
    train_models = x.FLAT_MODEL_CLEAN if train_models is None else train_models
    query_models = q.FLAT_MODEL_CLEAN if query_models is None else query_models
    if not train_models.index.equals(x.index) or not query_models.index.equals(q.index):
        raise ValueError("Raw model labels must align with the feature rows")
    train = model_group(train_models)
    query = model_group(query_models)
    levels = pd.Index(train.dropna().unique())
    column = "FLAT_MODEL_CLEAN" if variant == "grouped" else "FLAT_MODEL_GROUP"
    x[column] = pd.Categorical(train, categories=levels)
    q[column] = pd.Categorical(query.where(query.isin(levels)), categories=levels)
    return x, q
