import numpy as np
import numpy.testing as npt
from neuralnet.linalg import broadcaste
from neuralnet.test_linalg import (scopy_verify,
                                saxpy_verify,
                                sgemv_verify,
                                sgemm_verify)

def test_scopy():
   scopy_verify()

def test_saxpy():
    saxpy_verify()

def test_sgemv():
    sgemv_verify()

def test_sgemm():
    sgemm_verify()

def test_broadcaste():
    x = np.zeros((100, 10)).astype(np.float32)
    y = np.arange(10).astype(np.float32)

    expected = x + y
    broadcaste(x, y)

    npt.assert_almost_equal(expected, x)

