"""
privacy.py
----------
A minimal differential-privacy mechanism: additive Gaussian noise applied to
a client's model update before it leaves the hospital and is sent to the
server. This is the standard "DP-SGD"-style idea in simplified form -- it
does not compute a formal privacy budget (epsilon), but it demonstrates the
core mechanic and the accuracy trade-off it introduces, which is the point of
this project's privacy-utility evaluation.

Set DP_SIGMA = 0.0 to disable noise entirely (pure FedAvg baseline).
"""

import numpy as np


def add_dp_noise(weights, sigma: float, seed=None):
    """Adds i.i.d. Gaussian noise N(0, sigma^2) to every value in every
    weight array. Larger sigma => stronger privacy, lower accuracy.
    """
    if sigma <= 0:
        return weights
    rng = np.random.default_rng(seed)
    return [w + rng.normal(loc=0.0, scale=sigma, size=w.shape) for w in weights]
