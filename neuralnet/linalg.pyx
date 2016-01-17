#cython: boundscheck=False
#cython: cdivision=True
#cython: wraparound=False
from libc.string cimport memcpy

from neuralnet.core cimport float_array_1d_t, float_array_2d_t
from neuralnet cimport linalg

#
# vector operations
#

# x <- y
cdef void scopy(float_array_1d_t x, float_array_1d_t y) nogil:
    lib_scopy(x.shape[0], <float*>&x[0], 1, <float*>&y[0], 1)

# y += alpha * x
cdef void saxpy(float alpha, float_array_1d_t x, float_array_1d_t y) nogil:
    lib_saxpy(x.shape[0], alpha, <float*>&x[0], 1, <float*>&y[0], 1)

#
# matrix-vector single precision
#

#  A = alph * A x + beta * y
cdef void sgemv(float alpha, float_array_2d_t A, float_array_1d_t x, float beta, float_array_1d_t y) nogil:
    lib_sgemv(CblasRowMajor, CblasNoTrans, A.shape[0], A.shape[1], alpha,
            <float*>&A[0, 0], A.shape[1], <float*>&x[0], 1, beta, <float*>&y[0], 1)
#
# matrix-matrix single precision
#

# C = alpha * A B + beta * C
cdef void sgemm(float alpha, float_array_2d_t A, float_array_2d_t B,
                float beta, float_array_2d_t C) nogil:
    lib_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, C.shape[0], C.shape[1],
              B.shape[0], alpha, <float*>&A[0, 0], A.shape[1], <float*>&B[0, 0],
              B.shape[1], beta, <float*>&C[0, 0], C.shape[1])

#
# utility functions
#
cpdef int broadcaste(float_array_2d_t x, float_array_1d_t y) nogil except -1:
    cdef int n, m, i
    n = x.shape[0]
    m = y.shape[0]
    if x.shape[1] != m:
        with gil:
            raise ValueError('Arrays not broadcastable.')
    for i in range(n):
        memcpy(<float*>&x[0, 0] + i * m, <float*>&y[0], m * sizeof(float))
