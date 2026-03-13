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

EPOCHS = 20


def train(
    model: MLP,
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.05,
    decay_rate: float = 0.01,
) -> list[tuple[float, float]]:
    history = []
    for epoch in range(EPOCHS):
        scores = forward(model, X)
        total_loss, acc = loss(model, scores, y)
        print(f"loss={total_loss.data:.4f}, accuracy={acc * 100:.1f}%")
        history.append((total_loss.data, acc))

        model.zero_grad()
        total_loss.backward()

        step(model, learning_rate)
        learning_rate *= 1 - decay_rate

    return history


def forward(model: MLP, X: np.ndarray) -> list[Value]:
    """Run each sample through the model, return list of scores."""
    inputs = [(Value(x1), Value(x2)) for x1, x2 in X]
    return [model(x)[0] for x in inputs]


def loss(model: MLP, scores: list[Value], y: np.ndarray) -> tuple[Value, float]:
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


def plot_decision_boundary(model: MLP, X: np.ndarray, y: np.ndarray):
    """Plot model's decision boundary with data points overlaid.

    Creates a grid over the input space, classifies each point,
    and uses contourf to color regions. Saves to plots/decision_boundary.png.
    """
    # Create a mesh grid with some padding around the data
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))

    # Classify each grid point
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    zz = np.array([model([x1, x2])[0].data for x1, x2 in grid_points])
    zz = zz.reshape(xx.shape)

    # Plot colored regions and data points
    plt.figure()
    plt.contourf(xx, yy, zz, levels=[-1e9, 0, 1e9], colors=["#deebf7", "#fee0d2"], alpha=0.8)
    plt.contour(xx, yy, zz, levels=[0], colors="black", linewidths=1)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c="steelblue", edgecolors="k", s=30, label="+1")
    plt.scatter(X[y == -1, 0], X[y == -1, 1], c="salmon", edgecolors="k", s=30, label="-1")
    plt.legend()
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Decision Boundary")
    plt.savefig("plots/decision_boundary.png")
    plt.close()


if __name__ == "__main__":
    np.random.seed(1338)
    random.seed(1338)

    X, y = make_moons(n_samples=100, noise=0.1)
    y = y * 2 - 1  # Convert {0, 1} to {-1, 1}
    print(f"{X.shape=} ({X.min().item(), X.max().item()})")
    print(f"{y.shape=} ({y.min().item(), y.max().item()})")

    model = MLP(2, [16, 16, 1])

    history = train(model, X, y)
    plot_loss_curve(history)
    plot_accuracy_curve(history)
    plot_decision_boundary(model, X, y)
