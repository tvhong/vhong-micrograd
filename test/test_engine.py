import pytest
import torch
from micrograd.engine import Value


def test_init():
    a = Value(3.0)
    assert a.data == 3.0
    assert a.grad == 0
    assert a._prev == set()
    assert a._op == ""


def test_repr():
    a = Value(3.0)
    assert repr(a) == "Value(data=3.0, grad=0.0)"


def test_add():
    # micrograd
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    c.grad = 1.0
    c._backward()

    # pytorch
    at = torch.tensor(2.0, requires_grad=True)
    bt = torch.tensor(3.0, requires_grad=True)
    ct = at + bt
    ct.backward()

    assert c.data == ct.data.item()
    assert a.grad == at.grad.item()
    assert b.grad == bt.grad.item()


def test_mul():
    # micrograd
    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    c.grad = 1.0
    c._backward()

    # pytorch
    at = torch.tensor(2.0, requires_grad=True)
    bt = torch.tensor(3.0, requires_grad=True)
    ct = at * bt
    ct.backward()

    assert c.data == ct.data.item()
    assert a.grad == at.grad.item()
    assert b.grad == bt.grad.item()


def test_pow():
    # micrograd
    a = Value(2.0)
    b = a**3
    b.grad = 1.0
    b._backward()

    # pytorch
    at = torch.tensor(2.0, requires_grad=True)
    bt = at**3
    bt.backward()

    assert b.data == bt.data.item()
    assert a.grad == at.grad.item()


@pytest.mark.parametrize("val", [2.0, -2.0, 0.0])
def test_relu(val: float) -> None:
    # micrograd
    a = Value(val)
    b = a.relu()
    b.grad = 1.0
    b._backward()

    # pytorch
    at = torch.tensor(val, requires_grad=True)
    bt = at.relu()
    bt.backward()

    assert b.data == bt.data.item()
    assert a.grad == at.grad.item()


def test_backward():
    # micrograd
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    d = c * a
    d.backward()

    # pytorch
    at = torch.tensor(2.0, requires_grad=True)
    bt = torch.tensor(3.0, requires_grad=True)
    ct = at + bt
    dt = ct * at
    ct.retain_grad()
    dt.retain_grad()
    dt.backward()

    assert d.data == dt.data.item()
    assert a.grad == at.grad.item()
    assert b.grad == bt.grad.item()
    assert c.grad == ct.grad.item()
    assert d.grad == dt.grad.item()
