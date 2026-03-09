class Value:
    """stores a single scalar value and its gradient"""

    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other) -> "Value":
        if not isinstance(other, Value):
            other = Value(other)

        parent = Value(self.data + other.data, _children=(self, other), _op="+")

        def _backward():
            other.grad += parent.grad
            self.grad += parent.grad

        parent._backward = _backward

        return parent
