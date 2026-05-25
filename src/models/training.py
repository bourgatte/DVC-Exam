import os
import pickle

import pandas as pd
from xgboost import XGBRegressor

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"

def main():
    with open(os.path.join(MODELS_DIR, "best_params.pkl"), "rb") as f:
        best_params = pickle.load(f)

    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train_scaled.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv"))
    y_train = y_train.values.ravel()

    model = XGBRegressor(
        random_state=0,
        objective="reg:squarederror",
        **best_params,
    )
    model.fit(X_train, y_train)

    with open(os.path.join(MODELS_DIR, "xgb_model.pkl"), "wb") as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    main()