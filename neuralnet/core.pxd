ctypedef float[::1] float_array_1d_t
ctypedef float [:, ::1] float_array_2d_t
ctypedef void (*act_type)(float_array_2d_t) nogil
