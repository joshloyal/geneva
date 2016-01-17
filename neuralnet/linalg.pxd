from neuralnet.core cimport float_array_1d_t, float_array_2d_t

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


cdef void scopy(float_array_1d_t x, float_array_1d_t y) nogil
cdef void saxpy(float alpha, float_array_1d_t x, float_array_1d_t y) nogil
cdef void sgemv(float alpha, float_array_2d_t A, float_array_1d_t x, float beta, float_array_1d_t y) nogil
cdef void sgemm(float alpha, float_array_2d_t A, float_array_2d_t B, float beta, float_array_2d_t C) nogil
cpdef int broadcaste(float_array_2d_t x, float_array_1d_t y) nogil except -1
