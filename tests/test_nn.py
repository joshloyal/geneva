import numpy as np
import numpy.testing as npt
from neuralnet import Layer, ActivationType
from neuralnet import NeuralNet

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0.0, x)


def test_feed_forward_sigmoid():
    X = np.random.random((1000, 100)).astype(np.float32)
    layer = Layer(100, 3, ActivationType.sigmoid)
    out = layer.feed_forward(X)
    exp_out = sigmoid(np.dot(X, layer.weights[0]) + layer.weights[1])

    npt.assert_almost_equal(out, exp_out)


def test_feed_forward_relu():
    X = np.random.random((1000, 100)).astype(np.float32)
    layer = Layer(100, 3, ActivationType.relu)
    out = layer.feed_forward(X)
    exp_out = relu(np.dot(X, layer.weights[0]) + layer.weights[1])

    npt.assert_almost_equal(out, exp_out)

def test_nn_init():
    net = NeuralNet([1000, 100, 100, 3], activation=ActivationType.relu)
