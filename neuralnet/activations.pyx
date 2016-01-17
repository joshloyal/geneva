cdef void sigmoid(float_array_2d_t X) nogil:
    cdef int n, m
    n = X.shape[0]
    m = X.shape[1]
    for i in range(n):
        for j in range(m):
            X[i, j] = sigmoidf(X[i, j])

cdef void d_sigmoid(float_array_2d_t X) nogil:
    cdef int n, m
    n = X.shape[0]
    m = X.shape[1]
    for i in range(n):
        for j in range(m):
            X[i, j] = d_sigmoidf(X[i, j])

cdef void relu(float_array_2d_t X) nogil:
    cdef int n, m
    n = X.shape[0]
    m = X.shape[1]
    for i in range(n):
        for j in range(m):
            X[i, j] = reluf(X[i, j])

cdef void d_relu(float_array_2d_t X) nogil:
    cdef int n, m
    n = X.shape[0]
    m = X.shape[1]
    for i in range(n):
        for j in range(m):
            X[i, j] = d_reluf(X[i, j])
