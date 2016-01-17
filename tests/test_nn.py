import numpy as np
import numpy.testing as npt
from neuralnet import Layer

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def test_feed_forward():
    X = np.random.random((1000, 100)).astype(np.float32)
    layer = Layer(100, 3)
    out = layer.feed_forward(X)
    exp_out = sigmoid(np.dot(X, layer.weights[0]) + layer.weights[1])

    npt.assert_almost_equal(out, exp_out)
