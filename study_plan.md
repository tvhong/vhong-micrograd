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

- [ ] **Commit 11**: Add `nonlin` param to `Neuron` (skip relu on last layer) + `MLP` returns single `Value` when final layer has 1 output — test: linear neuron can output negative values, MLP(3,[4,1]) returns `Value` not list
- [ ] **Commit 12**: Training notebook setup — generate `make_moons` dataset, visualize it with matplotlib scatter plot, build MLP(2, [16, 16, 1])
- [ ] **Commit 13**: Loss function — implement SVM max-margin (hinge) loss + L2 regularization. Quiz: what is hinge loss, why L2 reg, what does `(1 + -yi*scorei).relu()` mean geometrically?
- [ ] **Commit 14**: Training loop — implement SGD with learning rate decay, train for 100 steps, print loss + accuracy each step, observe loss going down. Quiz: what is SGD, why zero_grad before backward, why decay learning rate?
- [ ] **Commit 15**: Visualization — plot loss curve over training steps, plot decision boundary (contour plot), understand what the model learned
- [ ] **Commit 16**: Integration test — automated test that trains MLP on make_moons, asserts loss decreases and accuracy > 90%

---

## Learning Strategy

### What I (Claude) implement

- **Commit 1**: Scaffold — boilerplate
- **Commit 2**: `Value.__init__` + `__repr__` — walk through the foundation
- **Commit 3**: `__add__` + test — full demonstration of the pattern: forward value, closure-based `_backward`, PyTorch comparison test

### What you implement (with guidance)

- **Commits 4–8**: `__mul__`, `__pow__`, `relu`, `backward()`, operator sugar — same pattern as `__add__`, only the math changes
- **Commits 9–10**: `nn.py` — composing `Value` ops into neural net modules

### Quizzes

Before each op you implement, you'll be quizzed on:

- The math: e.g. *"What is ∂(x*y)/∂x?"\*
- The pattern: e.g. _"What should `_backward` capture in its closure?"_
- Concepts: e.g. _"Why do we `+=` instead of `=` when accumulating gradients?"_

---

## Key Tips

1. **Math first, code second.** Derive the local gradient on paper before touching the keyboard. The code is just translating `∂out/∂self` into Python.

2. **The closure pattern is the key insight.** `_backward` captures `self`, `other`, and `out` — a mini-function that knows how to push gradients backward through one operation.

3. **`+=` not `=` for grads.** A single `Value` node can appear multiple times in a graph (e.g. `x * x`). Gradients from all uses must accumulate.

4. **Always verify against PyTorch.** Write the test first, then implement until it passes.

5. **Draw the graph.** For any expression, sketch the DAG. `backward()` is just a reverse post-order traversal of that graph.
