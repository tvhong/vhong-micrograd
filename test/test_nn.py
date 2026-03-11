from micrograd.nn import MLP, Layer, Module, Neuron
from micrograd.engine import Value


def test_module_parameters_empty():
    m = Module()
    assert m.parameters() == []


def test_neuron_parameters_count():
    n = Neuron(3)
    assert len(n.parameters()) == 4  # 3 weights + 1 bias


def test_parameters_not_zero():
    n = Neuron(3)
    assert all(
        p.data != 0 for p in n.parameters()
    )  # might fail occasionally but quite unlikely


def test_neuron_forward_returns_value():
    n = Neuron(3)
    out = n([1.0, 2.0, 3.0])
    assert hasattr(out, "data")
    assert hasattr(out, "grad")


def test_neuron_relu_applied():
    """Output should always be non-negative due to relu"""
    n = Neuron(3)
    for _ in range(100):  # run many times since weights are random
        out = n([1.0, -1.0, 0.5])
        assert out.data >= 0


def test_neuron_zero_grad():

    n = Neuron(2)
    # Force positive weights so pre-relu sum is positive and grads flow
    n._weights = [Value(1.0), Value(1.0)]
    n._bias = Value(0.5)

    out = n([2.0, 3.0])
    out.backward()

    # grads should be non-zero after backward
    assert any(p.grad != 0 for p in n.parameters())

    n.zero_grad()
    assert all(p.grad == 0 for p in n.parameters())


def test_neuron_nonlin_true():
    n = Neuron(1, nonlin=True)
    # Force negative weights to
    n._weights = [Value(-1.0)]
    n._bias = Value(-2.0)

    out = n([5.0])

    assert out.data == 0


def test_neuron_nonlin_false():
    n = Neuron(1, nonlin=False)
    # Force negative weights to
    n._weights = [Value(-1.0)]
    n._bias = Value(-2.0)

    out = n([5.0])

    assert out.data == -7  # 5 * -1 - 2


def test_layer_parameters_count():
    l1 = Layer(3, 4)

    assert len(l1.parameters()) == 4 * (3 + 1)


def test_layer_output_shape():
    out = Layer(3, 4)([1, 2, 3])
    assert len(out) == 4


def test_layer_output_non_negative():
    for _ in range(100):
        out = Layer(3, 4)([1, 2, 3])
        assert all(y.data >= 0 for y in out)


def test_mlp_parameters_count():
    mlp = MLP(3, [4, 4, 1])

    assert len(mlp.parameters()) == (3 + 1) * 4 + (4 + 1) * 4 + (4 + 1) * 1


def test_mlp_output_shape():
    out = MLP(3, [4, 4, 1])([1, 2, 3])

    assert len(out) == 1


def test_mlp_last_layer_is_linear():
    mlp = MLP(2, [2, 1])

    # Hidden layer (relu): both neurons output 1.0
    mlp._layers[0]._neurons[0]._weights = [Value(1.0), Value(1.0)]
    mlp._layers[0]._neurons[0]._bias = Value(0.0)
    mlp._layers[0]._neurons[1]._weights = [Value(1.0), Value(1.0)]
    mlp._layers[0]._neurons[1]._bias = Value(0.0)

    # Last layer (linear): w=[-1, -1], b=0 → -1*1 + -1*1 + 0 = -2
    mlp._layers[1]._neurons[0]._weights = [Value(-1.0), Value(-1.0)]
    mlp._layers[1]._neurons[0]._bias = Value(0.0)

    out = mlp([0.5, 0.5])
    assert out[0].data < 0  # negative output proves no relu on last layer


def test_layer_controlled_weights():
    layer = Layer(2, 2)

    # Neuron 0: w=[1, 2], b=0.5 → relu(1*3 + 2*4 + 0.5) = relu(11.5) = 11.5
    layer._neurons[0]._weights = [Value(1.0), Value(2.0)]
    layer._neurons[0]._bias = Value(0.5)

    # Neuron 1: w=[-1, -1], b=0.0 → relu(-1*3 + -1*4 + 0) = relu(-7) = 0
    layer._neurons[1]._weights = [Value(-1.0), Value(-1.0)]
    layer._neurons[1]._bias = Value(0.0)

    out = layer([3.0, 4.0])
    assert out[0].data == 11.5
    assert out[1].data == 0.0  # relu clamps negative to 0
