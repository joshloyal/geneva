#cython: boundscheck=False
#cython: cdivision=True
#cython nonecheck=False
from cython.parallel import prange

cimport numpy as np
import numpy as np

from cymem.cymem cimport Pool

from core cimport float_array_1d_t, float_array_2d_t, act_type
from linalg cimport sgemm, broadcaste
from activations cimport sigmoid, d_sigmoid, relu, d_relu

class ActivationType:
    sigmoid = 0
    relu = 1

cdef struct neural_net_layer:
    int n_inputs
    int n_outputs
    float_array_2d_t W  # weight matrix (n_inputs, n_outputs)
    float_array_1d_t b  # bias vector (n_outputs)
    float_array_2d_t act  # current activation (n_outputs)
    neural_net_layer* prev  # pointer to the previous layer
    neural_net_layer* next  # point to the next layer
    act_type activation
    act_type derivative
    float_array_2d_t delta_W
    float_array_1d_t delta_b

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

    def __cinit__(self, int n_inputs, int n_outputs, int activation_type):
        """ constructor """
        self.mem = Pool()
        self.layer = <neural_net_layer*>self.mem.alloc(1, sizeof(neural_net_layer))
        self.layer.n_inputs = n_inputs
        self.layer.n_outputs = n_outputs
        self.layer.W = np.ascontiguousarray(
            np.random.rand(n_inputs, n_outputs) * 0.1, dtype=np.float32)

        self.layer.b = np.ascontiguousarray(
            np.random.randn(n_outputs) * 0.1, dtype=np.float32)

        if activation_type == 0:
            self.layer.activation = sigmoid
            self.layer.derivative = d_sigmoid
        elif activation_type == 1:
            self.layer.activation = relu
            self.layer.derivative = d_relu

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

cdef class NeuralNet:
    cdef list layers
    cdef int n_features, n_outputs
    cdef int activation
    cdef Layer head, tail

    def __init__(self, shape, activation):
        self.layers = list(shape)
        self.n_features = self.layers[0]
        self.n_outputs = self.layers[-1]
        self.head = None
        self.tail = None
        self.activation = activation
        self.init()

    def init(self):
        self.head = Layer(self.layers[0], self.layers[1], self.activation)
        prev = self.head

        for k in xrange(1, len(self.layers) - 2):
            curr = Layer(self.layers[k], self.layers[k + 1], self.activation)
            curr.set_prev(prev)
            prev.set_next(curr)
            prev = curr

        self.tail = Layer(self.layers[-2], self.layers[-1], self.activation)  # change to output nonlinearity
        self.tail.set_prev(prev)
        prev.set_next(self.tail)

    def feed_forward(self):
        pass

# X = (100, 10),
# W = (10, 3)
# b = (10, 1)

cdef void feed_forward(neural_net_layer* layer, float_array_2d_t X) nogil:
    # act = X * W.T + b
    broadcaste(layer.act, layer.b)
    sgemm(1.0, X, layer.W, 1.0, layer.act)

    # apply the non-linearity (this memory access is slow)
    layer.activation(layer.act)

cdef void back_prop(neural_net_layer* layer, float_array_2d_t grad) nogil:
    pass
