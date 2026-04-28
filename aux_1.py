"""Utility functions for NASA C-MAPSS FD001 Remaining Useful Life prediction.

The FD001 files contain rows with:
engine_id, cycle, 3 operational settings, and 21 sensor readings.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

ID_COL = "engine_id"
CYCLE_COL = "cycle"
SETTING_COLS = [f"setting_{i}" for i in range(1, 4)]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
ALL_COLUMNS = [ID_COL, CYCLE_COL] + SETTING_COLS + SENSOR_COLS

DEFAULT_DROP_COLS = [
    "setting_3", "sensor_1", "sensor_5", "sensor_6", "sensor_10",
    "sensor_16", "sensor_18", "sensor_19"
]


def load_fd001(path: str | Path) -> pd.DataFrame:
    """Load a whitespace-delimited C-MAPSS FD001 file."""
    df = pd.read_csv(path, sep=r"\s+", header=None)
    if df.shape[1] != len(ALL_COLUMNS):
        raise ValueError(f"Expected {len(ALL_COLUMNS)} columns, got {df.shape[1]}")
    df.columns = ALL_COLUMNS
    return df


def add_train_rul(df: pd.DataFrame, cap: int | None = 130) -> pd.DataFrame:
    """Add RUL labels to the training data using max cycle per engine."""
    out = df.copy()
    max_cycle = out.groupby(ID_COL)[CYCLE_COL].transform("max")
    out["RUL_raw"] = max_cycle - out[CYCLE_COL]
    out["RUL"] = out["RUL_raw"].clip(upper=cap) if cap is not None else out["RUL_raw"]
    return out


def latest_cycle_rows(test_df: pd.DataFrame) -> pd.DataFrame:
    """Return the final available cycle for each test engine."""
    idx = test_df.groupby(ID_COL)[CYCLE_COL].idxmax()
    return test_df.loc[idx].sort_values(ID_COL).reset_index(drop=True)


def prepare_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Select model features in a consistent order."""
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return df[feature_cols].copy()


def evaluate_regression(y_true, y_pred) -> Dict[str, float]:
    """Return standard regression metrics."""
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def save_artifact(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_artifact(path: str | Path):
    return joblib.load(path)
