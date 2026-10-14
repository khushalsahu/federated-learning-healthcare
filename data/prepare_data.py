"""
prepare_data.py
----------------
Loads a medical dataset and splits it into:
  - A held-out global TEST set (used to evaluate the federated / centralised models fairly)
  - N "hospital" partitions (simulated clients), each saved as its own CSV,
    mimicking data silos that never get pooled together.

Dataset used by default: scikit-learn's Breast Cancer Wisconsin dataset
(569 patients, 30 diagnostic features, binary label: malignant / benign).
It ships with scikit-learn, so no internet download is required.

To use a different medical CSV (e.g. Pima Indians Diabetes or the UCI Heart
Disease dataset), drop the file in this folder and edit `load_raw_dataset()`
below to read it with pandas instead -- the rest of the pipeline (partitioning,
client training, aggregation) does not need to change.
"""

import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
NUM_HOSPITALS = 3
TEST_SIZE = 0.2
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_raw_dataset() -> pd.DataFrame:
    """Loads the raw dataset as a single DataFrame with a 'target' column.

    Swap this function's body to load a different CSV, e.g.:
        df = pd.read_csv("diabetes.csv")
        df = df.rename(columns={"Outcome": "target"})
        return df
    """
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    # 'target' column already present: 0 = malignant, 1 = benign
    return df


def partition_iid(df: pd.DataFrame, n_parts: int, seed: int = RANDOM_SEED):
    """Shuffles and splits the dataframe into n roughly-equal, non-overlapping
    partitions (IID split -- each hospital sees a similar class balance).
    Real hospitals would likely have skewed (non-IID) distributions; that is
    a natural extension noted in the project README.
    """
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    indices = np.array_split(np.arange(len(shuffled)), n_parts)
    return [shuffled.iloc[idx].reset_index(drop=True) for idx in indices]


def main():
    df = load_raw_dataset()
    print(f"Loaded dataset with {len(df)} rows and {df.shape[1] - 1} features.")

    # Held-out global test set: simulates an independent evaluation set that
    # NO hospital or the server has trained on. Used to fairly compare the
    # federated global model against the centralised baseline.
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=df["target"]
    )

    test_path = os.path.join(OUT_DIR, "test.csv")
    test_df.to_csv(test_path, index=False)
    print(f"Saved held-out test set ({len(test_df)} rows) -> {test_path}")

    # Partition the remaining training data across simulated hospitals.
    parts = partition_iid(train_df, NUM_HOSPITALS)
    hospital_names = [f"hospital_{chr(65 + i)}" for i in range(NUM_HOSPITALS)]  # A, B, C...

    for name, part in zip(hospital_names, parts):
        path = os.path.join(OUT_DIR, f"{name}.csv")
        part.to_csv(path, index=False)
        print(f"Saved {name} ({len(part)} rows, "
              f"class balance {part['target'].value_counts().to_dict()}) -> {path}")

    print("\nDone. Each hospital_*.csv is a private, non-shared partition.")
    print("The federated learning pipeline will only ever read its own hospital's file.")


if __name__ == "__main__":
    main()
