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

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons

from micrograd.engine import Value
from micrograd.nn import MLP

# --- 1. Generate dataset ---
np.random.seed(1338)
random.seed(1338)

X: np.ndarray
y: np.ndarray
X, y = make_moons(n_samples=100, noise=0.1)
y = y * 2 - 1  # Convert {0, 1} to {-1, 1}
# print(f"{X=}, {y=}")
print(f"{X.shape=} ({X.min().item(), X.max().item()})")
print(f"{y.shape=} ({y.min().item(), y.max().item()})")

# --- 2. Build model ---
model = MLP(2, [16, 16, 1])


EPOCHS = 20


def train(
    X: np.ndarray, y: np.ndarray, learning_rate: float = 0.05, decay_rate: float = 0.01
) -> list[tuple[float, float]]:
    losses = []
    for epoch in range(EPOCHS):
        scores = forward(X)
        total_loss, acc = loss(scores, y)
        print(f"loss={total_loss.data:.4f}, accuracy={acc * 100:.1f}%")
        losses.append((total_loss.data, acc))

        model.zero_grad()
        total_loss.backward()

        step(model, learning_rate)
        learning_rate *= 1 - decay_rate

    return losses


def forward(X: np.ndarray) -> list[Value]:
    """Run each sample through the model, return list of scores."""
    inputs = [(Value(x1), Value(x2)) for x1, x2 in X]
    return [model(x)[0] for x in inputs]


def loss(scores: list[Value], y: np.ndarray) -> tuple[Value, float]:
    """Compute hinge loss + L2 reg and accuracy from pre-computed scores.

    Returns:
        (total_loss, accuracy) where total_loss is a Value (for backprop)
        and accuracy is a float between 0 and 1.
    """
    # Hinge loss — penalise scores with margin < 1
    losses = [_hinge_loss(yi, scorei) for yi, scorei in zip(y.tolist(), scores)]
    data_loss = sum(losses) / len(losses)

    # L2 regularization — penalise large weights
    alpha = 1e-4
    reg_loss = alpha * sum([p * p for p in model.parameters()])

    total_loss = data_loss + reg_loss

    accuracy = _accuracy(y, scores)

    return total_loss, accuracy


def step(model: MLP, learning_rate: float):
    for p in model.parameters():
        p.data -= learning_rate * p.grad


def _hinge_loss(yi: float, scorei: Value) -> Value:
    return (1 + -yi * scorei).relu()


def _accuracy(y: np.ndarray, scores: list[Value]) -> float:
    assert y.shape[0] == len(scores)
    num_correct = 0
    for yi, scorei in zip(y.tolist(), scores):
        num_correct += int((yi > 0) == (scorei.data > 0))

    return num_correct / y.shape[0]


def plot_loss_curve(history: list[tuple[float, float]]):
    """Plot loss over epochs and save to plots/loss_curve.png."""
    losses = [h[0] for h in history]
    plt.figure()
    plt.plot(range(len(losses)), losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.savefig("plots/loss_curve.png")
    plt.close()


def plot_accuracy_curve(history: list[tuple[float, float]]):
    """Plot accuracy over epochs and save to plots/accuracy_curve.png."""
    accuracies = [h[1] * 100 for h in history]
    plt.figure()
    plt.plot(range(len(accuracies)), accuracies)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training Accuracy")
    plt.savefig("plots/accuracy_curve.png")
    plt.close()


if __name__ == "__main__":
    history = train(X, y)
    plot_loss_curve(history)
    plot_accuracy_curve(history)
