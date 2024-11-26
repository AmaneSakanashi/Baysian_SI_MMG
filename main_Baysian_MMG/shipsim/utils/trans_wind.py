import numpy as np


def polarTrue2polarApparent(U_T, gamma_T, u, vm, psi):
    wA_x, wA_y = polarTrue2xyApparent(U_T, gamma_T, u, vm, psi)
    U_A = np.sqrt(wA_x**2 + wA_y**2)
    gamma_A = np.arctan2(wA_y, wA_x)
    return U_A, gamma_A


def polarTrue2xyApparent(U_T, gamma_T, u, vm, psi):
    wT_x = U_T * np.cos(gamma_T - psi)
    wT_y = U_T * np.sin(gamma_T - psi)
    return xyTrue2xyApparent(wT_x, wT_y, u, vm)


def xyTrue2xyApparent(wT_x, wT_y, u, vm):
    wA_x = wT_x - u
    wA_y = wT_y - vm
    return wA_x, wA_y
