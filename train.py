"""
Train an MLP on the make_moons dataset using micrograd.

Plan:
  Commit 12 - Setup & loss function:
    1. Generate make_moons dataset (100 samples, noise=0.1), labels as -1/+1
    2. Build MLP(2, [16, 16, 1])
    3. Implement loss() that computes hinge loss + L2 reg + accuracy

  Commit 13 - Training loop:
    4. SGD with learning rate decay for 100 steps
    5. Print loss + accuracy each step

  Commit 14 - Visualization:
    6. Plot loss curve over training steps
    7. Plot decision boundary (contour plot)
    8. Save plots to plots/
"""

import random

import numpy as np
from sklearn.datasets import make_moons

from micrograd.engine import Value
from micrograd.nn import MLP

# --- 1. Generate dataset ---
np.random.seed(1337)
random.seed(1337)

X: np.ndarray
y: np.ndarray
X, y = make_moons(n_samples=100, noise=0.1)
y = y * 2 - 1  # Convert {0, 1} to {-1, 1}
# print(f"{X=}, {y=}")
print(f"{X.shape=} ({X.min().item(), X.max().item()})")
print(f"{y.shape=} ({y.min().item(), y.max().item()})")

# --- 2. Build model ---
model = MLP(2, [16, 16, 1])


def loss() -> tuple[Value, float]:
    """Compute total loss and accuracy over the full dataset.

    Returns:
        (total_loss, accuracy) where total_loss is a Value (for backprop)
        and accuracy is a float between 0 and 1.

    Steps:
        1. Forward pass: run each sample through the model
        2. Hinge loss: for each sample, compute (1 + -yi * scorei).relu()
        3. Data loss: average the hinge losses
        4. L2 regularization: alpha * sum(p * p) for all parameters, alpha = 1e-4
        5. Total loss: data_loss + reg_loss
        6. Accuracy: fraction of samples where sign(score) matches label
    """
    # Forward pass — get a score for each input
    inputs = ...  # TODO: convert each row of X into a list of Values
    scores = ...  # TODO: run each input through the model

    # Hinge loss — penalise scores with margin < 1
    losses = ...  # TODO: (1 + -yi * scorei).relu() for each sample
    data_loss = ...  # TODO: average the losses

    # L2 regularization — penalise large weights
    alpha = 1e-4
    reg_loss = ...  # TODO: alpha * sum of p*p for all model parameters

    total_loss = ...  # TODO: data_loss + reg_loss

    # Accuracy
    accuracy = ...  # TODO: fraction of samples where (yi > 0) == (scorei.data > 0)

    return total_loss, accuracy


if __name__ == "__main__":
    total_loss, acc = loss()
    print(f"loss={total_loss.data:.4f}, accuracy={acc * 100:.1f}%")
