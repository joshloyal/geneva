from core cimport float_array_2d_t

cdef extern from "math.h":
    cdef float expf(float x) nogil

cdef inline float sigmoidf(float x) nogil:
    return 1.0 / (1 + expf(-x))

cdef inline float d_sigmoidf(float x) nogil:
    return x * (1.0 - x)

cdef inline float reluf(float x) nogil:
    if x < 0.0:
        return 0.0
    return x

cdef inline float d_reluf(float x) nogil:
    if x < 0.0:
        return 0.0
    elif x > 0.0:
        return 1.0
    else:
        return 0.1

cdef void sigmoid(float_array_2d_t X) nogil
cdef void d_sigmoid(float_array_2d_t X) nogil
cdef void relu(float_array_2d_t X) nogil
cdef void d_relu(float_array_2d_t X) nogil
