import os

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

RAW_PATH = "data/raw/raw.csv"
OUTPUT_DIR = "data/processed"


def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["split"]

    df = pd.read_csv(RAW_PATH)

    X = df.drop(columns=["date","silica_concentrate"])
    y = df["silica_concentrate"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params["test_size"],
        random_state=params["random_state"],
    )

    X_train.to_csv(os.path.join(OUTPUT_DIR, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(OUTPUT_DIR, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(OUTPUT_DIR, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(OUTPUT_DIR, "y_test.csv"), index=False)

if __name__ == "__main__":
    main()
