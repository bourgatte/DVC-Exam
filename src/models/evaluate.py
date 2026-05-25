import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"
METRICS_DIR = "metrics"
DATA_DIR = "data"


def main():
    with open(os.path.join(MODELS_DIR, "xgb_model.pkl"), "rb") as f:
        model = pickle.load(f)

    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test_scaled.csv"))
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv"))
    y_true = y_test.values.ravel()

    y_pred = model.predict(X_test)

    predictions = X_test.copy()
    predictions["silica_concentrate_true"] = y_true
    predictions["silica_concentrate_pred"] = y_pred
    predictions.to_csv(os.path.join(DATA_DIR, "prediction.csv"), index=False)

    mse = mean_squared_error(y_true, y_pred)
    scores = {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }

    with open(os.path.join(METRICS_DIR, "scores.json"), "w") as f:
        json.dump(scores, f, indent=4)

if __name__ == "__main__":
    main()
