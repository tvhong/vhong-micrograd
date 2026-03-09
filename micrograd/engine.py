class Value:
    """stores a single scalar value and its gradient"""

    def __init__(self, data, children=(), op=""):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(children)
        self._op = op

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other) -> "Value":
        if not isinstance(other, Value):
            other = Value(other)

        parent = Value(self.data + other.data, children=(self, other), op="+")

        def _backward():
            other.grad += parent.grad
            self.grad += parent.grad

        parent._backward = _backward

        return parent
