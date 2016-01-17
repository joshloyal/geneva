#cython: boundscheck=False
#cython: cdivision=True
#cython nonecheck=False
from libc.string cimport memcpy

from cython.parallel import prange

cimport numpy as np
import numpy as np

from cymem.cymem cimport Pool

from linalg cimport linalg
from linalg import linalg

cdef extern from "math.h":
    cdef float expf(float x) nogil

cdef class Layer:
    cdef Pool mem
    cdef neural_net_layer* layer
    cdef Layer prev, next

    property shape:
        def __get__(self):
            return self.layer.n_inputs, self.layer.n_outputs

    property weights:
        def __get__(self):
            return np.asarray(self.layer.W), np.asarray(self.layer.b)

    def __cinit__(self, int n_inputs, int n_outputs):
        """ constructor """
        self.mem = Pool()
        self.layer = <neural_net_layer*>self.mem.alloc(1, sizeof(neural_net_layer))
        self.layer.n_inputs = n_inputs
        self.layer.n_outputs = n_outputs
        self.layer.W = np.ascontiguousarray(
            np.random.rand(n_inputs, n_outputs) * 0.1, dtype=np.float32)

        self.layer.b = np.ascontiguousarray(
            np.random.randn(n_outputs) * 0.1, dtype=np.float32)

    cdef void set_prev(self, Layer prev):
        self.prev = prev
        self.layer.prev = prev.layer

    cdef void set_next(self, Layer next):
        self.next = next
        self.layer.next = next.layer

    cdef void set_batch(self, int batch_size):
        self.layer.act = np.ascontiguousarray(
            np.zeros((batch_size, self.layer.n_outputs)), dtype=np.float32)

    def feed_forward(self, X):
        self.set_batch(X.shape[0])
        feed_forward(self.layer, X)
        return np.asarray(self.layer.act)


cdef struct neural_net_layer:
    int n_inputs
    int n_outputs
    float [:, ::1] W  # weight matrix (n_inputs, n_outputs)
    float [::1] b  # bias vector (n_outputs)
    float [:, ::1] act  # current activation (n_outputs)
    neural_net_layer* prev  # pointer to the previous layer
    neural_net_layer* next  # point to the next layer
    #float [:, ::1] delta_W
    #float [:, ::1] delta_b
    #float [::1] prev_activations
    #float [::1] activations

# X = (100, 10),
# W = (10, 3)
# b = (10, 1)

cpdef int broadcaste(float [:, ::1] x, float [:] y) nogil except -1:
    cdef int n, m, i
    n = x.shape[0]
    m = y.shape[0]
    if x.shape[1] != m:
        with gil:
            raise ValueError('Arrays not broadcastable.')
    for i in range(n):
        memcpy(<float*>&x[0, 0] + i * m, <float*>&y[0], m * sizeof(float))


cdef void feed_forward(neural_net_layer* layer, float [:, ::1] X) nogil:
    # act = X * W.T + b
    broadcaste(layer.act, layer.b)
    linalg.sgemm(1.0, X, layer.W, 1.0, layer.act)

    # apply the non-linearity (this memory access is slow)
    cdef int n, m
    n = layer.act.shape[0]
    m = layer.act.shape[1]
    for i in range(n):
        for j in range(m):
            layer.act[i, j] = 1.0 / (1 + expf(-layer.act[i, j]))

