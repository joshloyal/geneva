import numpy as np
import numpy.testing as npt
from neuralnet import broadcaste

def test_broadcaste():
    x = np.zeros((100, 10)).astype(np.float32)
    y = np.arange(10).astype(np.float32)

    expected = x + y
    broadcaste(x, y)

    npt.assert_almost_equal(expected, x)

