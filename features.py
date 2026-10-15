"""Feature engineering and preprocessing shared by training, API, dashboard and Lambda."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"], errors="coerce")
    if ts.isna().any():
        raise ValueError("Some timestamps could not be parsed (use ISO format, e.g. 2025-03-01T14:30:00).")
    out["hour"] = ts.dt.hour
    out["day_of_week"] = ts.dt.dayofweek
    out["is_night"] = (out["hour"] <= 5).astype(int)
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["log_amount"] = np.log1p(out["amount"].astype(float))
    return out


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), C.NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), C.CATEGORICAL_FEATURES),
    ])


def clean_names(names):
    return [n.replace("num__", "").replace("cat__", "") for n in names]
