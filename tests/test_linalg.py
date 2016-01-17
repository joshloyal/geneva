import numpy as np
import numpy.testing as npt
from linalg import ddot, sdot
from linalg.test_linalg import scopy_verify, saxpy_verify, sgemv_verify

dot = {np.float32: sdot, np.float64: ddot}

def dot_comparison(dtype):
    np.random.seed(1234)
    A = np.random.random((100, 100)).astype(dtype)
    B = np.random.random((100, 100)).astype(dtype)

    numpy_dot = np.dot(A, B)
    geneva_dot = dot[dtype](A, B)

    npt.assert_almost_equal(numpy_dot, geneva_dot)


def test_dot_single_precision():
    dot_comparison(dtype=np.float32)


def test_dot_double_precision():
    dot_comparison(dtype=np.float64)

def test_scopy():
   scopy_verify()

def test_saxpy():
    saxpy_verify()

def test_sgemv():
    sgemv_verify()
