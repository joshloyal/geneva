#cython: boundscheck=False
#cython: cdivision=True
#cython: wraparound=False

import_array()

#
# vector operations
#
cdef void scopy(np.ndarray x, np.ndarray y) nogil:
    lib_scopy(x.shape[0], <float*>x.data, 1, <float*>y.data, 1)

cdef void saxpy(float alpha, np.ndarray x, np.ndarray y) nogil:
    lib_saxpy(x.shape[0], alpha, <float*>x.data, 1, <float*>y.data, 1)

#
# matrix-vector single precision
#
cdef void sgemv(float alpha, np.ndarray A, np.ndarray x, float beta, np.ndarray y) nogil:
    lib_sgemv(CblasRowMajor, CblasNoTrans, A.shape[0], A.shape[1], alpha,
            <float*>A.data, A.shape[1], <float*>x.data, 1, beta, <float*>y.data, 1)
#
# matrix-matrix single precision
#
cdef void sgemm_(float alpha, float [:, ::1] A, float [:, ::1] B,
                 float beta, float [:, ::1] C) nogil:
    lib_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, C.shape[0], C.shape[1],
              B.shape[0], alpha, <float*>&A[0, 0], A.shape[1], <float*>&B[0, 0],
              B.shape[1], beta, <float*>&C[0, 0], C.shape[1])


cdef void sgemm3(np.ndarray A, np.ndarray B, np.ndarray C) nogil:
    """ inplace matrix multiplication """
    sgemm_(1.0, A, B, 0.0, C)

cdef np.ndarray sgemm(np.ndarray A, np.ndarray B):
    cdef np.ndarray C = smnewempty(A.shape[0], B.shape[1])
    sgemm_(1.0, A, B, 0.0, C)
    return C

cpdef np.ndarray sdot(np.ndarray A, np.ndarray B):
    return sgemm(A, B)

#
# matrix-matrix double precision
#
cdef void dgemm_(double alpha, np.ndarray A, np.ndarray B,
                 double beta, np.ndarray C) nogil:
    lib_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, C.shape[0], C.shape[1],
              B.shape[0], alpha, <double*>A.data, A.shape[1], <double*>B.data,
              B.shape[1], beta, <double*>C.data, C.shape[1])


cdef void dgemm3(np.ndarray A, np.ndarray B, np.ndarray C) nogil:
    """ inplace matrix multiplication """
    dgemm_(1.0, A, B, 0.0, C)

cdef np.ndarray dgemm(np.ndarray A, np.ndarray B):
    cdef np.ndarray C = dmnewempty(A.shape[0], B.shape[1])
    dgemm_(1.0, A, B, 0.0, C)
    return C

cpdef np.ndarray ddot(np.ndarray A, np.ndarray B):
    return dgemm(A, B)

#
# Utility functions
#
cdef np.ndarray svnewempty(int N):
    cdef np.npy_intp length[1]
    length[0] = N
    Py_INCREF(np.NPY_FLOAT)
    return PyArray_EMPTY(1, length, np.NPY_FLOAT, 0)

cdef np.ndarray dmnewempty(int M, int N):
    cdef np.npy_intp length[2]
    length[0] = M
    length[1] = N
    Py_INCREF(np.NPY_DOUBLE)
    return PyArray_EMPTY(2, length, np.NPY_DOUBLE, 0)

cdef np.ndarray smnewempty(int M, int N):
    cdef np.npy_intp length[2]
    length[0] = M
    length[1] = N
    Py_INCREF(np.NPY_FLOAT)
    return PyArray_EMPTY(2, length, np.NPY_FLOAT, 0)
