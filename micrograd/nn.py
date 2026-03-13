import json
import random
from collections.abc import Sequence
from typing import Any, override

from micrograd.engine import Value


class Module:
    def parameters(self) -> list[Value]:
        return []

    def zero_grad(self):
        for param in self.parameters():
            param.grad = 0


class Neuron(Module):
    def __init__(self, n_inputs: int, nonlin: bool = True) -> None:
        self._weights: list[Value] = []
        for i in range(n_inputs):
            self._weights.append(self._sample())

        self._bias: Value = self._sample()
        self._nonlin: bool = nonlin

    def __call__(self, input: Sequence[float | Value]) -> Value:
        assert len(input) == len(self._weights), (
            "Number input elements must match configured inputs"
        )

        raw = sum((w * x for w, x in zip(input, self._weights))) + self._bias
        return raw.relu() if self._nonlin else raw

    @override
    def parameters(self) -> list[Value]:
        return self._weights + [self._bias]

    def _sample(self, low: float = -1, high: float = 1) -> Value:
        return Value(random.uniform(low, high))


class Layer(Module):
    def __init__(self, nin: int, nout: int, nonlin: bool = True) -> None:
        self._shape: tuple[int, int] = (nin, nout)
        self._neurons: list[Neuron] = [Neuron(nin, nonlin) for _ in range(nout)]

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
        self._nin: int = nin
        self._dims: list[int] = dims
        all_dims = [nin] + dims
        assert len(all_dims) >= 3, "MLP must have at least 1 hidden layer"
        self._layers: list[Layer] = []
        for i in range(len(all_dims) - 2):
            nin = all_dims[i]
            nout = all_dims[i + 1]
            self._layers.append(Layer(nin, nout, nonlin=True))

        self._layers.append(Layer(all_dims[-2], all_dims[-1], nonlin=False))

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

    def save(self, path: str) -> None:
        data: dict[str, Any] = {
            "nin": self._nin,
            "dims": self._dims,
            "layers": [
                {
                    "neurons": [
                        {
                            "weights": [w.data for w in neuron._weights],
                            "bias": neuron._bias.data,
                        }
                        for neuron in layer._neurons
                    ]
                }
                for layer in self._layers
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "MLP":
        with open(path) as f:
            data = json.load(f)

        model = cls(data["nin"], data["dims"])
        for layer, layer_data in zip(model._layers, data["layers"]):
            for neuron, neuron_data in zip(layer._neurons, layer_data["neurons"]):
                for w, w_val in zip(neuron._weights, neuron_data["weights"]):
                    w.data = w_val
                neuron._bias.data = neuron_data["bias"]

        return model
