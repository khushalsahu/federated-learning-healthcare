"""
baseline_centralized.py
-------------------------
Trains the SAME model architecture (SGDClassifier logistic regression) on the
POOLED training data from all hospitals combined -- i.e. the traditional,
non-private approach where all patient data is centralised on one server.

This is used purely as a benchmark: comparing its accuracy against the
federated model (federated_simulation.py) tells us how much accuracy, if any,
is sacrificed by keeping each hospital's data private instead of pooling it.

Usage:
    python baseline_centralized.py --epochs 15
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_utils import build_model, init_model, CLASSES

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def main():
    parser = argparse.ArgumentParser(description="Train the centralised baseline model.")
    parser.add_argument("--epochs", type=int, default=15,
                         help="training epochs, matched to the FL --rounds count for a fair comparison")
    args = parser.parse_args()

    hospital_files = sorted(glob.glob(os.path.join(DATA_DIR, "hospital_*.csv")))
    pooled = pd.concat([pd.read_csv(f) for f in hospital_files], ignore_index=True)
    print(f"Pooled {len(pooled)} patient records from {len(hospital_files)} hospitals "
          f"into a single centralised training set (this is what FL avoids doing).")

    X_train = pooled.drop(columns=["target"]).values
    y_train = pooled["target"].values

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    X_test = scaler.transform(test_df.drop(columns=["target"]).values)
    y_test = test_df["target"].values

    model = build_model()
    init_model(model, X_train_scaled[:2], y_train[:2])

    rows = []
    for epoch in range(1, args.epochs + 1):
        model.partial_fit(X_train_scaled, y_train)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)
        acc = accuracy_score(y_test, preds)
        loss = log_loss(y_test, probs, labels=CLASSES)

        rows.append({"epoch": epoch, "test_accuracy": acc, "test_loss": loss})
        print(f"Epoch {epoch:2d}/{args.epochs} | Test accuracy: {acc:.4f} | Test loss: {loss:.4f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "baseline_metrics.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nSaved baseline metrics -> {out_path}")
    print(f"Final centralised baseline — test accuracy: {rows[-1]['test_accuracy']:.4f}, "
          f"test loss: {rows[-1]['test_loss']:.4f}")


if __name__ == "__main__":
    main()
