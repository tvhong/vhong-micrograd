import random

from micrograd.engine import Value


class Module:
    def __init__(self, n_inputs: int) -> None:
        self._weights: list[Value] = []
        for i in range(n_inputs):
            self._weights.append(self._sample())

        self._bias: Value = self._sample()

    def __call__(self, input: list[float]) -> Value:
        assert len(input) == len(self._weights), (
            "Number input elements must match configured inputs"
        )

        return (sum((w * x for w, x in zip(input, self._weights))) + self._bias).relu()

    def parameters(self) -> list[Value]:
        return self._weights + [self._bias]

    def zero_grad(self):
        for param in self.parameters():
            param.grad = 0

    def _sample(self, low: float = -1, high: float = 1) -> Value:
        return Value(random.uniform(low, high))
