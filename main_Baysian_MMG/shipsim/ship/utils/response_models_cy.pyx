# cython: language_level=3
from libc.math cimport tanh

cdef double linear_delay_ode_rhs(double y, double u, double K):
    """Right hand side of ordinary differential equations for linear delay systems

    Args:
        y (double): Output variables
        u (double): Input variables
        K (double): Constant rate of output variables
    """
    cdef double dt = 0.1
    cdef double dydt = K * tanh((u - y) / (K * dt))
    return dydt
