import torch
from micrograd.engine import Value


def test_init():
    a = Value(3.0)
    assert a.data == 3.0
    assert a.grad == 0
    assert a._prev == set()
    assert a._op == ''


def test_repr():
    a = Value(3.0)
    assert repr(a) == "Value(data=3.0, grad=0)"
