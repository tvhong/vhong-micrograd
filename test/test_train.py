import random

import numpy as np
from sklearn.datasets import make_moons

from micrograd.nn import MLP
from train import train


def test_training_reduces_loss_and_reaches_90_accuracy():
    """Train MLP on make_moons end-to-end and verify it learns."""
    np.random.seed(1337)
    random.seed(1337)

    X, y = make_moons(n_samples=100, noise=0.1)
    y = y * 2 - 1

    model = MLP(2, [16, 16, 1])
    history = train(model, X, y, learning_rate=1.0, decay_rate=0.009, epochs=100)

    first_loss = history[0][0]
    final_loss = history[-1][0]
    final_acc = history[-1][1]

    assert final_loss < first_loss, f"Loss did not decrease: {first_loss} -> {final_loss}"
    assert final_acc >= 0.9, f"Accuracy too low: {final_acc * 100:.1f}%"
