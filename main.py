"""Train and evaluate machine learning models for FD001 RUL prediction.

Run:
    python main.py

Expected data layout:
    data/train_FD001.txt
    data/test_FD001.txt
    data/RUL_FD001.txt
"""
from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from aux_1 import (
    DEFAULT_DROP_COLS,
    ID_COL,
    CYCLE_COL,
    SETTING_COLS,
    SENSOR_COLS,
    add_train_rul,
    evaluate_regression,
    latest_cycle_rows,
    load_fd001,
)

RUL_CAP = 130
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
FIG_DIR = BASE_DIR / "figures"
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)


def main() -> None:
    train_path = DATA_DIR / "train_FD001.txt"
    test_path = DATA_DIR / "test_FD001.txt"
    rul_path = DATA_DIR / "RUL_FD001.txt"

    train = add_train_rul(load_fd001(train_path), cap=RUL_CAP)
    test = load_fd001(test_path)
    true_rul = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["RUL_raw"])
    true_rul["engine_id"] = np.arange(1, len(true_rul) + 1)
    true_rul["RUL"] = true_rul["RUL_raw"].clip(upper=RUL_CAP)

    feature_cols = [c for c in SETTING_COLS + SENSOR_COLS if c not in DEFAULT_DROP_COLS]
    X_train = train[feature_cols]
    y_train = train["RUL"]

    test_last = latest_cycle_rows(test)
    X_test = test_last[feature_cols]
    y_test = true_rul["RUL"]
    y_test_raw = true_rul["RUL_raw"]

    models = {
        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),
        "Random Forest": RandomForestRegressor(
            n_estimators=80,
            max_depth=12,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
    }

    rows = []
    predictions = pd.DataFrame({"engine_id": true_rul["engine_id"], "actual_RUL": y_test, "actual_RUL_raw": y_test_raw})

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = np.clip(model.predict(X_test), 0, RUL_CAP)
        metrics = evaluate_regression(y_test, preds)
        raw_metrics = evaluate_regression(y_test_raw, preds)
        rows.append({
            "model": name,
            "RMSE_capped": metrics["RMSE"],
            "MAE_capped": metrics["MAE"],
            "R2_capped": metrics["R2"],
            "RMSE_raw_labels": raw_metrics["RMSE"],
            "MAE_raw_labels": raw_metrics["MAE"],
        })
        predictions[f"pred_{name.replace(' ', '_').lower()}"] = preds

    results = pd.DataFrame(rows).sort_values("RMSE_capped")
    results.to_csv(REPORT_DIR / "model_results.csv", index=False)
    predictions.to_csv(REPORT_DIR / "test_predictions.csv", index=False)

    best_name = results.iloc[0]["model"]
    best_model = models[best_name]
    joblib.dump({
        "model": best_model,
        "feature_cols": feature_cols,
        "rul_cap": RUL_CAP,
        "drop_cols": DEFAULT_DROP_COLS,
        "best_model_name": best_name,
        "results": results,
    }, MODEL_DIR / "best_rul_model.joblib")

    # Plot model comparison
    plt.figure(figsize=(7, 4.2))
    plt.bar(results["model"], results["RMSE_capped"])
    plt.ylabel("RMSE (cycles, capped RUL)")
    plt.title("Model comparison on FD001 test set")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "model_comparison.png", dpi=220)
    plt.close()

    # Plot actual vs predicted for best model
    best_pred_col = f"pred_{best_name.replace(' ', '_').lower()}"
    plt.figure(figsize=(6.2, 4.2))
    plt.scatter(predictions["actual_RUL"], predictions[best_pred_col], alpha=0.8)
    plt.plot([0, RUL_CAP], [0, RUL_CAP], linestyle="--")
    plt.xlabel("Actual RUL")
    plt.ylabel("Predicted RUL")
    plt.title(f"Actual vs predicted RUL ({best_name})")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "actual_vs_predicted.png", dpi=220)
    plt.close()

    # Feature importance when available
    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(10)
        importances.to_csv(REPORT_DIR / "feature_importance.csv")
        plt.figure(figsize=(7, 4.2))
        importances.sort_values().plot(kind="barh")
        plt.xlabel("Importance")
        plt.title(f"Top features ({best_name})")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "feature_importance.png", dpi=220)
        plt.close()

    print("Training complete.")
    print(results.to_string(index=False))
    print(f"Best model: {best_name}")
    print(f"Saved model to {MODEL_DIR / 'best_rul_model.joblib'}")


if __name__ == "__main__":
    main()
