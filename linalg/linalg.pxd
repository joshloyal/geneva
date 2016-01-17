cimport numpy as np

cdef extern from "Python.h":

    cdef void Py_INCREF(object)


cdef extern from "numpy/arrayobject.h":

    cdef void import_array()

    cdef object PyArray_EMPTY(int nd, np.npy_intp *dims, int typenum, int fortran)

    int NPY_DOUBLE
    int NPY_FLOAT

cdef extern from "cblas.h":

    enum CBLAS_ORDER: CblasRowMajor, CblasColMajor
    enum CBLAS_TRANSPOSE: CblasNoTrans, CblasTrans, CblasConjTrans


    ###########################################################################
    # BLAS level 1 routines (i.e. vector - vector operations)
    ###########################################################################
    # vector copy y <- x
    void lib_scopy "cblas_scopy"(int N, float *x, int dx, float *y, int dy) nogil

    # vector addition y += alpha*x
    void lib_saxpy "cblas_saxpy"(int N, float alpha, float *x, int dx,
                                 float *y, int dx) nogil

    ###########################################################################
    # BLAS level 2 routines (i.e. matrix - vector products)
    ###########################################################################
    # A = alpha * A x + beta * y
    void lib_sgemv "cblas_sgemv"(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA,
                                 int M, int N, float alpha, float *A, int lda,
                                 float *x, int dx, float beta, float *y, int dy) nogil

    ###########################################################################
    # BLAS level 3 routines (i.e. matrix - matrix products)
    ###########################################################################

    ### C = alpha * A B + beta * C

    # float32 (single precision)
    void lib_sgemm "cblas_sgemm"(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA,
                                 CBLAS_TRANSPOSE TransB, int M, int N, int K,
                                 float alpha, float *A, int lda, float *B,
                                 int ldb, float beta, float *C, int ldc) nogil

    # float64 (double precision)
    void lib_dgemm "cblas_dgemm"(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA,
                                 CBLAS_TRANSPOSE TransB, int M, int N, int K,
                                 double alpha, double *A, int lda, double *B,
                                 int ldb, double beta, double *C, int ldc) nogil


cdef void scopy(np.ndarray x, np.ndarray y) nogil
cdef void saxpy(float alpha, np.ndarray x, np.ndarray y) nogil
cdef void sgemv(float  alpha, np.ndarray A, np.ndarray x, float beta, np.ndarray y) nogil

cdef np.ndarray svnewempty(int N)
cdef void sgemm_(float alpha, float [:, ::1] A, float [:, ::1] B, float beta, float [:, ::1] C) nogil
cdef void sgemm3(np.ndarray A, np.ndarray B, np.ndarray C) nogil
cdef np.ndarray sgemm(np.ndarray A, np.ndarray B)
cpdef np.ndarray sdot(np.ndarray A, np.ndarray B)
cdef np.ndarray smnewempty(int M, int N)

cdef void dgemm_(double alpha, np.ndarray A, np.ndarray B, double beta, np.ndarray C) nogil
cdef void dgemm3(np.ndarray A, np.ndarray B, np.ndarray C) nogil
cdef np.ndarray dgemm(np.ndarray A, np.ndarray B)
cpdef np.ndarray ddot(np.ndarray A, np.ndarray B)
cdef np.ndarray dmnewempty(int M, int N)
