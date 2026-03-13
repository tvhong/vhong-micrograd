# Study Plan: Implementing Backprop (micrograd)

## Implementation Plan

### Phase 1: Project scaffold

- [x] **Commit 1**: Set up package structure + test infrastructure (`micrograd/__init__.py`, `engine.py`, `nn.py`, `test/test_engine.py`, `setup.py`)

### Phase 2: `Value` class — engine + tests

- [x] **Commit 2**: `Value.__init__` + `__repr__` — test: instantiate, check `.data`, `.grad`, `repr`
- [x] **Commit 3**: `__add__` + backward — test: forward value correct, grad correct vs PyTorch
- [x] **Commit 4**: `__mul__` + backward — test: forward + grad vs PyTorch
- [x] **Commit 5**: `__pow__` + backward — test: forward + grad vs PyTorch
- [x] **Commit 6**: `relu` + backward — test: positive/negative input, grad vs PyTorch
- [x] **Commit 7**: `backward()` (topo sort + chain rule) — test: multi-node graph, all grads vs PyTorch
- [x] **Commit 8**: Operator sugar (`__neg__`, `__radd__`, `__sub__`, `__rsub__`, `__rmul__`, `__truediv__`, `__rtruediv__`) — test: full `test_more_ops` expression vs PyTorch

### Phase 3: Neural net modules + tests

- [x] **Commit 9**: `Module` + `Neuron` — test: forward pass output shape, `parameters()` count, `zero_grad()`
- [x] **Commit 10**: `Layer` + `MLP` — test: MLP forward pass, backward pass updates gradients correctly

### Phase 4: Training end-to-end

- [x] **Commit 11**: Add `nonlin` param to `Neuron` (skip relu on last layer) — test: linear neuron can output negative values, MLP last layer is linear
- [x] **Commit 12**: Training script scaffold (`train.py`) — generate `make_moons` dataset, build MLP(2, [16, 16, 1]), implement loss function (SVM hinge loss + L2 reg). Quiz: what is hinge loss, why L2 reg, what does `(1 + -yi*scorei).relu()` mean geometrically?
- [x] **Commit 13**: Training loop in `train.py` — implement SGD with learning rate decay, train for 100 steps, print loss + accuracy each step. Quiz: what is SGD, why zero_grad before backward, why decay learning rate?
- [x] **Commit 14**: Visualization in `train.py` — plot loss curve over training steps, plot decision boundary (contour plot), save plots to `plots/`. Understand what the model learned.
- [x] **Commit 15**: Integration test — automated pytest that trains MLP on make_moons, asserts loss decreases and accuracy > 90%
- [x] **Commit 16**: Save/load for MLP — persist model weights and architecture to disk, reload without retraining

---

## Learning Strategy

### Quizzes

Before each op you implement, you'll be quizzed on:

- The math: e.g. *"What is ∂(x*y)/∂x?"\*
- The pattern: e.g. _"What should `_backward` capture in its closure?"_
- Concepts: e.g. _"Why do we `+=` instead of `=` when accumulating gradients?"_
