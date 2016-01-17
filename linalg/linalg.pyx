#cython: boundscheck=False
#cython: cdivision=True
#cython: wraparound=False

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
