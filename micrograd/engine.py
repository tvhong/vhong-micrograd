from __future__ import annotations

from typing import Any, Callable


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

    def __pow__(self, pow: float) -> "Value":
        out = Value(self.data**pow, (self,), "**")

        def _backward() -> None:
            self.grad += pow * self.data ** (pow - 1) * out.grad

        out._backward = _backward

        return out

    def relu(self) -> "Value":
        out_val = self.data if self.data > 0 else 0
        out = Value(out_val, (self,), "relu")

        def _backward() -> None:
            self.grad += out.grad if self.data > 0 else 0

        out._backward = _backward

        return out

    def backward(self) -> None:
        visited: set["Value"] = set()
        topo: list["Value"] = list()

        def build_topo(v: "Value"):
            """Build a topology from children to root"""
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)  # append after children

        build_topo(self)
        topo = topo[::-1]

        self.grad = 1
        for v in topo:
            v._backward()

    def __neg__(self) -> "Value":
        return self * -1

    def __sub__(self, other: "Value") -> "Value":
        return self + -other

    def __radd__(self, other: Any) -> "Value":
        return self + other

    def __rmul__(self, other: Any) -> "Value":
        return self * other
