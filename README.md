# micrograd

A from-scratch implementation of [Karpathy's micrograd](https://github.com/karpathy/micrograd) — a tiny autograd engine and neural network library, built as a learning exercise.

## What's in here

- **`micrograd/engine.py`** — `Value` class: a scalar with automatic differentiation. Supports `+`, `*`, `**`, `relu`, and full `backward()` via reverse-mode autodiff (topological sort + chain rule).
- **`micrograd/nn.py`** — `Neuron`, `Layer`, `MLP` classes built on top of `Value`. The last layer is linear (no relu) so the network can output negative scores.
- **`train.py`** — Trains an MLP(2, [16, 16, 1]) on sklearn's `make_moons` dataset using SVM hinge loss + L2 regularization. Generates loss curve, accuracy curve, and decision boundary plots.

## Setup

```bash
uv sync
```

## Usage

Train the model and generate plots:

```bash
uv run python train.py
```

Plots are saved to `plots/` (loss curve, accuracy curve, decision boundary).

## Tests

Run fast tests:

```bash
uv run pytest
```

Run slow integration test (trains full model, ~2 min):

```bash
uv run pytest -m slow
```

Run all tests:

```bash
uv run pytest -m ''
```
