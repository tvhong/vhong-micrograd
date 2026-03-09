from __future__ import annotations

from typing import Callable


class Value:
    """stores a single scalar value and its gradient"""

    def __init__(
        self, data: float, children: tuple["Value", ...] = (), op: str = ""
    ) -> None:
        self.data: float = data
        self.grad: float = 0.0
        self._backward: Callable[[], None] = lambda: None
        self._prev: set["Value"] = set(children)
        self._op: str = op

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other: "Value | float") -> "Value":
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            other.grad += out.grad
            self.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other: "Value | float") -> "Value":
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            other.grad += self.data * out.grad
            self.grad += other.data * out.grad

        out._backward = _backward

        return out
