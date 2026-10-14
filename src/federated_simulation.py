"""
federated_simulation.py
------------------------
Runs the full Federated Learning experiment described in the project synopsis:

    Global Server -> Hospital A, B, C -> Local Training -> Model Weight
    Sharing -> Federated Averaging -> Global Model -> Disease Prediction

For each communication round:
  1. The current global model weights are broadcast to every hospital client.
  2. Each hospital trains locally on its own private data only
     (HospitalClient.fit) and returns its updated weights.
  3. The server aggregates all client weights via Federated Averaging
     (FedAvg) to produce the new global model.
  4. The global model is evaluated on a held-out test set that no hospital
     has ever trained on.

This is a single-process simulation: all "hospitals" run in the same Python
process for simplicity and reproducibility. Each HospitalClient object only
ever touches its own CSV file, so the *logical* privacy boundary (no raw data
leaves the hospital) is real and enforced by the code structure, even though
everything runs on one machine. See README.md for how this maps to a genuine
multi-machine deployment using Docker/Flower's networked mode.

Usage:
    python federated_simulation.py                  # plain FedAvg
    python federated_simulation.py --rounds 15
    python federated_simulation.py --dp-sigma 0.05   # with differential privacy noise
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
from model_utils import build_model, init_model, get_weights, set_weights, average_weights, CLASSES
from fl_client import HospitalClient

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def load_scaler():
    """Fits a StandardScaler on the *union* of hospital feature distributions
    only (never on the held-out test set), simulating an agreed-upon
    preprocessing standard hospitals adopt before joining the federation.
    """
    hospital_files = sorted(glob.glob(os.path.join(DATA_DIR, "hospital_*.csv")))
    frames = [pd.read_csv(f) for f in hospital_files]
    combined = pd.concat(frames, ignore_index=True)
    X = combined.drop(columns=["target"]).values
    scaler = StandardScaler().fit(X)
    return scaler, hospital_files


def load_test_set(scaler):
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    X_test = scaler.transform(test_df.drop(columns=["target"]).values)
    y_test = test_df["target"].values
    return X_test, y_test


def evaluate_global_model(weights, X_test, y_test):
    model = build_model()
    init_model(model, X_test[:2], y_test[:2])
    set_weights(model, weights)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)
    acc = accuracy_score(y_test, preds)
    loss = log_loss(y_test, probs, labels=CLASSES)
    return acc, loss


def run_federated_training(num_rounds: int, local_epochs: int, dp_sigma: float):
    scaler, hospital_files = load_scaler()
    X_test, y_test = load_test_set(scaler)

    clients = [
        HospitalClient(
            client_id=os.path.basename(f).replace(".csv", ""),
            csv_path=f,
            scaler=scaler,
            local_epochs=local_epochs,
            dp_sigma=dp_sigma,
        )
        for f in hospital_files
    ]
    print(f"Initialised {len(clients)} simulated hospitals: "
          f"{[c.client_id for c in clients]}")
    if dp_sigma > 0:
        print(f"Differential privacy ENABLED (sigma={dp_sigma}) -- "
              f"Gaussian noise will be added to every client update.")

    # Global model starts at zero weights (round 0), matches each client's init.
    global_weights = get_weights(clients[0].model)

    rows = []
    for rnd in range(1, num_rounds + 1):
        client_weights, client_examples = [], []
        round_client_metrics = []

        for client in clients:
            updated_weights, n_examples, fit_info = client.fit(global_weights, config={})
            client_weights.append(updated_weights)
            client_examples.append(n_examples)

            loss, n_eval, eval_info = client.evaluate(updated_weights, config={})
            round_client_metrics.append({
                "round": rnd,
                "client_id": client.client_id,
                "local_accuracy": eval_info["accuracy"],
                "local_loss": loss,
            })

        # --- Federated Averaging on the server ---
        global_weights = average_weights(client_weights, client_examples)

        global_acc, global_loss = evaluate_global_model(global_weights, X_test, y_test)

        for m in round_client_metrics:
            m["global_test_accuracy"] = global_acc
            m["global_test_loss"] = global_loss
            m["dp_sigma"] = dp_sigma
            rows.append(m)

        print(f"Round {rnd:2d}/{num_rounds} | "
              f"Global test accuracy: {global_acc:.4f} | "
              f"Global test loss: {global_loss:.4f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "fl_metrics.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nSaved per-round metrics -> {out_path}")

    weights_path = os.path.join(RESULTS_DIR, "global_model_weights.npz")
    np.savez(weights_path, coef=global_weights[0], intercept=global_weights[1])
    print(f"Saved final global model weights -> {weights_path}")

    final_acc, final_loss = evaluate_global_model(global_weights, X_test, y_test)
    print(f"\nFinal federated global model — test accuracy: {final_acc:.4f}, "
          f"test loss: {final_loss:.4f}")
    return final_acc, final_loss


def main():
    parser = argparse.ArgumentParser(description="Run the Federated Learning simulation.")
    parser.add_argument("--rounds", type=int, default=15, help="number of FL communication rounds")
    parser.add_argument("--local-epochs", type=int, default=1, help="local training epochs per round per hospital")
    parser.add_argument("--dp-sigma", type=float, default=0.0,
                         help="stddev of Gaussian noise added to client updates (0 = disabled)")
    args = parser.parse_args()

    run_federated_training(args.rounds, args.local_epochs, args.dp_sigma)


if __name__ == "__main__":
    main()
