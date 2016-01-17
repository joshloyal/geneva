import time as time

cimport linalg
cimport numpy as np

import numpy as np
import numpy.testing as npt

speed_base = 200000
test_sizes = [4, 15, 30, 50]


def scopy_verify():
    x = np.array(np.random.random(4), dtype=np.float32)
    y = np.array(np.random.random(4), dtype=np.float32)
    linalg.scopy(x, y)
    npt.assert_almost_equal(x, y)


def saxpy_verify():
    x = np.array( np.random.random( (5) ),   dtype=np.float32 )
    y = np.array( np.random.random( (5) ),   dtype=np.float32 )
    temp = 1.2 * x + y
    linalg.saxpy( 1.2, x, y )
    npt.assert_almost_equal(temp, y)


def sgemv_verify():
    A = np.array( np.random.random( (4,5) ), dtype=np.float32 )
    x = np.array( np.random.random( (5) ),   dtype=np.float32 )
    y = np.array( np.random.random( (4) ),   dtype=np.float32 )

    cdef np.ndarray[float, ndim=2, mode='c'] A_
    cdef np.ndarray[float, ndim=1, mode='c'] x_, y_
    A_ = A; x_ = x; y_ = y

    # y = 1.2 * A x + 2.1 * y
    temp = 1.2*np.dot(A,x) + 2.1*y
    linalg.sgemv(1.2, A, x, 2.1, y)
    npt.assert_almost_equal(temp, y)


cdef sgemv_speed( int size ):

    cdef int i, loops

    loops = speed_base*10/(<int>(size**1.2))

    A = np.array( np.random.random( (size,size) ), dtype=np.float32 )
    x = np.array( np.random.random( (size) ),      dtype=np.float32 )
    y = np.array( np.random.random( (size) ),      dtype=np.float32 )

    cdef np.ndarray[float, ndim=2, mode='c'] A_
    cdef np.ndarray[float, ndim=1, mode='c'] x_, y_
    A_ = A; x_ = x; y_ = y

    print "numpy.dot +: ",
    start = time.clock()
    for i in range(loops):
        y *= 2.1
        y += 1.2*np.dot(A,x)
    np_rate = loops/(time.clock()-start)
    print "%9.0f kc/s" % (np_rate/1000)

    loops *= 3

    print "sgemv:      ",
    start = time.clock()
    for i in range(loops):
        linalg.sgemv( 1.2, A_, x_, 2.1, y_ )
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

cdef sgemm_speed(int size):
    cdef int i, loops

    loops = speed_base * 150 / (size * size)

    X = np.array( np.random.random( (size, size) ), dtype=np.float32)
    Y = np.array( np.random.random( (size, size) ), dtype=np.float32)
    Z = np.empty((size, size), dtype=np.float32)

    cdef float [:, ::1] X_, Y_, Z_
    X_ = X; Y_ = Y; Z_ = Z

    print "size = %r" % size
    print "numpy.dot: ",
    start = time.clock()
    for i in range(loops): np.dot( X, Y )
    np_rate = loops/(time.clock()-start)
    print "%9.0f kc/s" % (np_rate/1000)

    print "linalg sgemm:     ",
    start = time.clock()
    for i in range(loops):
        linalg.sgemm( X, Y )
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "linalg sgemm3:     ",
    start = time.clock()
    for i in range(loops):
        linalg.sgemm3(X, Y, Z)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

cdef dgemm_speed(int size):
    cdef int i, loops
    loops = speed_base * 150 / (size * size)

    X = np.array( np.random.random( (size, size) ), dtype=np.float64)
    Y = np.array( np.random.random( (size, size) ), dtype=np.float64)
    Z = np.empty((size, size), dtype=np.float64)

    cdef double [:, ::1] X_, Y_, Z_
    X_ = X; Y_ = Y; Z_ = Z

    print "size = %r" % size
    print "numpy.dot: ",
    start = time.clock()
    for i in range(loops): np.dot( X, Y )
    np_rate = loops/(time.clock()-start)
    print "%9.0f kc/s" % (np_rate/1000)

    print "linalg dgemm:     ",
    start = time.clock()
    for i in range(loops):
        linalg.dgemm( X, Y )
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "linalg dgemm3:     ",
    start = time.clock()
    for i in range(loops):
        linalg.dgemm3(X, Y, Z)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

def test_linalg():
    print "Test single precision gemv"
    for size in test_sizes:
        sgemv_speed(size); print

    print "Test single precision gemm"
    for size in test_sizes:
        sgemm_speed(size); print

    print "Test double precision gemm"
    for size in test_sizes:
        dgemm_speed(size); print
