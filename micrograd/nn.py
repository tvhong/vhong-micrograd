import random
from collections.abc import Sequence
from typing import override

from micrograd.engine import Value


class Module:
    def parameters(self) -> list[Value]:
        return []

    def zero_grad(self):
        for param in self.parameters():
            param.grad = 0


class Neuron(Module):
    def __init__(self, n_inputs: int) -> None:
        self._weights: list[Value] = []
        for i in range(n_inputs):
            self._weights.append(self._sample())

        self._bias: Value = self._sample()

    def __call__(self, input: Sequence[float | Value]) -> Value:
        assert len(input) == len(self._weights), (
            "Number input elements must match configured inputs"
        )

        return (sum((w * x for w, x in zip(input, self._weights))) + self._bias).relu()

    @override
    def parameters(self) -> list[Value]:
        return self._weights + [self._bias]

    def _sample(self, low: float = -1, high: float = 1) -> Value:
        return Value(random.uniform(low, high))


class Layer(Module):
    def __init__(self, nin: int, nout: int) -> None:
        self._shape: tuple[int, int] = (nin, nout)
        self._neurons: list[Neuron] = [Neuron(nin) for _ in range(nout)]

    def __call__(self, input: Sequence[float | Value]) -> list[Value]:
        out: list[Value] = []
        for n in self._neurons:
            out.append(n(input))

        return out

    @override
    def parameters(self) -> list[Value]:
        params: list[Value] = []
        for n in self._neurons:
            params.extend(n.parameters())

        return params


class MLP(Module):
    def __init__(self, nin: int, dims: list[int]) -> None:
        nins = [nin] + dims[:-1]
        nouts = dims
        self._layers: list[Layer] = [Layer(nin, nout) for nin, nout in zip(nins, nouts)]

    def __call__(self, input: Sequence[float | Value]) -> list[Value]:
        x: list[Value] = [
            val if isinstance(val, Value) else Value(val) for val in input
        ]
        for layer in self._layers:
            x = layer(x)

        return x

    @override
    def parameters(self) -> list[Value]:
        params: list[Value] = []
        for layer in self._layers:
            params.extend(layer.parameters())
        return params
