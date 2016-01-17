import timeit

from linalg cimport scopy, saxpy, sgemv, sgemm
from linalg cimport float_array_1d_t, float_array_2d_t
cimport numpy as np

import numpy as np
import numpy.testing as npt

loops = 100000
test_sizes = [4, 15, 30, 50]


def scopy_verify():
    x = np.array(np.random.random(4), dtype=np.float32)
    y = np.array(np.random.random(4), dtype=np.float32)

    cdef float_array_1d_t x_, y_
    x_ = x; y_ = y

    scopy(x, y)
    npt.assert_almost_equal(x, y)

def saxpy_verify():
    x = np.array( np.random.random(5), dtype=np.float32 )
    y = np.array( np.random.random(5), dtype=np.float32 )

    cdef float_array_1d_t x_, y_
    x_ = x; y_ = y

    temp = 1.2 * x + y
    saxpy(1.2, x_, y_)
    npt.assert_almost_equal(temp, y)


def sgemv_verify():
    A = np.array( np.random.random((4, 5)), dtype=np.float32 )
    x = np.array( np.random.random(5), dtype=np.float32 )
    y = np.array( np.random.random(4), dtype=np.float32 )

    cdef float_array_2d_t  A_
    cdef float_array_1d_t  x_, y_
    A_ = A; x_ = x; y_ = y

    # y = 1.2 * A x + 2.1 * y
    temp = 1.2*np.dot(A, x) + 2.1*y
    sgemv(1.2, A_, x_, 2.1, y_)
    npt.assert_almost_equal(temp, y)

def sgemm_verify():
    X = np.array(np.random.random((3, 4)), dtype=np.float32)
    Y = np.array(np.random.random((4, 5)), dtype=np.float32)
    Z = np.array(np.random.random((3, 5)), dtype=np.float32)

    cdef float_array_2d_t X_, Y_, Z_
    X_ = X; Y_ = Y; Z_ = Z

    result = 2.3 * np.dot(X, Y) + 1.2 * Z
    sgemm(2.3, X_, Y_, 1.2, Z_)
    npt.assert_almost_equal(result, Z)


cdef sgemv_speed( int size ):
    A = np.array(np.random.random((size, size)), dtype=np.float32)
    x = np.array(np.random.random(size), dtype=np.float32)
    y = np.array(np.random.random(size), dtype=np.float32)

    cdef float_array_2d_t A_
    cdef float_array_1d_t  x_, y_
    A_ = A; x_ = x; y_ = y

    def numpy_dot(A, x, y):
        y = np.dot(A, x)

    numpy_t = timeit.Timer(lambda: numpy_dot(A, x, y))
    np_rate = loops / min(numpy_t.repeat(3, loops))
    print "numpy dot:\t %9.0f kc/s" % (np_rate/1000)

    blas_t = timeit.Timer(lambda: sgemv(1.0, A_, x_, 0.0, y_))
    blas_rate = loops / min(numpy_t.repeat(3, loops))
    print "blas sgemv:\t %9.0f kc/s %5.1fx" % (blas_rate/1000,blas_rate/np_rate)


cdef sgemm_speed(int size):
    X = np.array( np.random.random( (size, size) ), dtype=np.float32)
    Y = np.array( np.random.random( (size, size) ), dtype=np.float32)
    Z = np.empty((size, size), dtype=np.float32)

    cdef float_array_2d_t X_, Y_, Z_
    X_ = X; Y_ = Y; Z_ = Z

    def numpy_dot(X, Y, Z):
        Z = np.dot(X, Y)

    numpy_t = timeit.Timer(lambda: numpy_dot(X, Y, Z))
    np_rate = loops / min(numpy_t.repeat(3, loops))
    print "numpy dot:\t %9.0f kc/s" % (np_rate/1000)

    blas_t = timeit.Timer(lambda: sgemm(1.0, X_, Y_, 0.0, Z_))
    blas_rate = loops / min(numpy_t.repeat(3, loops))
    print "blas sgemm:\t %9.0f kc/s %5.1fx" % (blas_rate/1000,blas_rate/np_rate)

def test_linalg():
    print "Test single precision level 2 BLAS"
    for size in test_sizes:
        sgemv_speed(size); print

    print "Test single precision level 3 BLAS"
    for size in test_sizes:
        sgemm_speed(size); print

