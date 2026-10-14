"""
model_utils.py
---------------
A small logistic-regression model (scikit-learn's SGDClassifier) whose
weights can be extracted / replaced as plain NumPy arrays. This is what lets
it plug into Flower: Flower's FedAvg strategy only needs to know how to
average lists of NumPy arrays -- it doesn't care what framework produced them.
"""

import numpy as np
from sklearn.linear_model import SGDClassifier

N_FEATURES = 30       # matches the Breast Cancer Wisconsin dataset
CLASSES = np.array([0, 1])


def build_model() -> SGDClassifier:
    """Creates a fresh logistic-regression model trained via SGD.
    warm_start=True lets us call partial_fit repeatedly across rounds
    without resetting the learned weights each time.
    """
    return SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        learning_rate="optimal",
        warm_start=True,
        random_state=42,
    )


def init_model(model: SGDClassifier, X_sample: np.ndarray, y_sample: np.ndarray) -> None:
    """SGDClassifier only allocates coef_/intercept_ after the first fit call.
    We do one throwaway partial_fit so the parameter arrays exist and can be
    overwritten with the server's global weights before real training starts.
    """
    model.partial_fit(X_sample, y_sample, classes=CLASSES)
    model.coef_ = np.zeros_like(model.coef_)
    model.intercept_ = np.zeros_like(model.intercept_)


def get_weights(model: SGDClassifier):
    """Returns model parameters as a list of NumPy arrays: [coef_, intercept_]."""
    return [model.coef_.copy(), model.intercept_.copy()]


def set_weights(model: SGDClassifier, weights) -> None:
    """Loads a list of NumPy arrays [coef_, intercept_] into the model."""
    model.coef_ = weights[0].copy()
    model.intercept_ = weights[1].copy()


def average_weights(weights_list, num_examples_list):
    """Federated Averaging (FedAvg): a weighted average of client weights,
    weighted by how many local training examples each client used.
    This is the same core operation Flower's built-in FedAvg strategy performs.
    """
    total = sum(num_examples_list)
    avg_coef = sum(w[0] * n for w, n in zip(weights_list, num_examples_list)) / total
    avg_intercept = sum(w[1] * n for w, n in zip(weights_list, num_examples_list)) / total
    return [avg_coef, avg_intercept]
