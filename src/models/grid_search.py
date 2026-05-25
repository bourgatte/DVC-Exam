import os
import pickle

import pandas as pd
import yaml
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"


def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["gridsearch"]

    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train_scaled.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv"))
    y_train = y_train.values.ravel()

    model = XGBRegressor(random_state=42, objective="reg:squarederror")

    grid = GridSearchCV(
        estimator=model,
        param_grid=params["param_grid"],
        cv=params["cv"],
        scoring=params["scoring"],
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)

    with open(os.path.join(MODELS_DIR, "best_params.pkl"), "wb") as f:
        pickle.dump(grid.best_params_, f)

if __name__ == "__main__":
    main()
