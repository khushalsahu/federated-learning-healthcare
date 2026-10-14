"""
fl_client.py
------------
Defines HospitalClient: a Flower NumPyClient that trains only on its own
hospital's private data. This class is used by both:
  - federated_simulation.py (a manual, single-process FedAvg loop -- the
    default way to run this project, no extra infrastructure required), and
  - it is written to be fully compatible with Flower's official
    `flwr.simulation.start_simulation` / `flwr run` deployment path too,
    for anyone who wants to extend this into a real multi-process or
    multi-machine deployment (see README "Extending to Real Deployment").
"""

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import StandardScaler

import flwr as fl

from model_utils import build_model, init_model, get_weights, set_weights, CLASSES
from privacy import add_dp_noise


class HospitalClient(fl.client.NumPyClient):
    """Represents one simulated hospital. Owns its own data and its own
    local model instance; the server never sees `X_train` / `y_train`.
    """

    def __init__(self, client_id: str, csv_path: str, scaler: StandardScaler,
                 local_epochs: int = 1, dp_sigma: float = 0.0):
        self.client_id = client_id
        self.local_epochs = local_epochs
        self.dp_sigma = dp_sigma

        df = pd.read_csv(csv_path)
        X = df.drop(columns=["target"]).values
        y = df["target"].values

        # Each hospital fits/uses a shared, pre-fitted scaler (fit on training
        # distribution only) -- in a stricter setup each hospital could scale
        # locally instead; kept simple here for clarity.
        self.X = scaler.transform(X)
        self.y = y

        self.model = build_model()
        init_model(self.model, self.X[:2], self.y[:2])

    # --- Flower NumPyClient interface -------------------------------------

    def get_parameters(self, config):
        return get_weights(self.model)

    def fit(self, parameters, config):
        """Called each round: receive the global model, train locally,
        return the updated weights (optionally noised for privacy)."""
        set_weights(self.model, parameters)

        for _ in range(self.local_epochs):
            self.model.partial_fit(self.X, self.y)

        updated_weights = get_weights(self.model)
        if self.dp_sigma > 0:
            updated_weights = add_dp_noise(updated_weights, self.dp_sigma)

        metrics = {"client_id": self.client_id}
        return updated_weights, len(self.X), metrics

    def evaluate(self, parameters, config):
        """Called to score the current global model on this hospital's
        local data (used for per-hospital diagnostics in the dashboard)."""
        set_weights(self.model, parameters)
        preds_proba = self.model.predict_proba(self.X)
        preds = self.model.predict(self.X)

        loss = log_loss(self.y, preds_proba, labels=CLASSES)
        acc = accuracy_score(self.y, preds)

        return loss, len(self.X), {"accuracy": acc, "client_id": self.client_id}
