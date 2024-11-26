import numpy as np


def rotation(pos, psi):
    A = np.array([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]])
    return np.dot(pos, A.T)


def ship_coo(eta, Lpp, B, Lrate=0.6):
    pos_x = np.array(
        [-0.5 * Lpp, Lrate * 0.5 * Lpp, 0.5 * Lpp, Lrate * 0.5 * Lpp, -0.5 * Lpp]
    )
    pos_y = np.array([-0.5 * B, -0.5 * B, 0.0, 0.5 * B, 0.5 * B])
    points = np.concatenate([pos_x[:, np.newaxis], pos_y[:, np.newaxis]], axis=1)
    return rotation(points, eta[2]) + eta[0:2]


def rectangle_coo(eta, Lpp, B):
    pos_x = np.array([-0.5 * Lpp, 0.5 * Lpp, 0.5 * Lpp, -0.5 * Lpp])
    pos_y = np.array([-0.5 * B, -0.5 * B, 0.5 * B, 0.5 * B])
    points = np.concatenate([pos_x[:, np.newaxis], pos_y[:, np.newaxis]], axis=1)
    return rotation(points, eta[2]) + eta[0:2]


def ship_domain_miyauchi_coo(eta, nu, Lpp, B, W, scale, split_num=10, split_type=0):
    """This function returns the polygon that approximates the ship domain proposed by Miyauchi (2022)

    Args:
        eta (numpy.ndarray): position and headding angle (x_0, y_0, psi)
        nu (numpy.ndarray): velocity and angle velocity (u, v_m, r)
        Lpp (float): ship length
        B (float): ship breadth
        W (float):  the minimum passage width
        scale (float): scale compared to real ship
        split_num (int, optional): Defaults to 10.
        split_type (int, optional): Defaults to 0.

    Returns:
        numpy.ndarray: ship shape polygon

    Note:
        Yoshiki Miyauchi et al.
        Optimization on planning of trajectory and control of autonomous berthing and unberthing for the realistic port geometry,
        Ocean Engineering, Volume 245, 2022, 110390, ISSN 0029-8018, https://doi.org/10.1016/j.oceaneng.2021.110390.
    """
    kt = (1852 / 3600) * np.sqrt(scale)
    U_min = 1 * kt
    U_max = 6 * kt
    U = np.linalg.norm(nu[0:2])
    U = np.max([U, U_min])
    U = np.min([U, U_max])
    #
    Lx_min = 0.75 * Lpp
    Ly_min = B
    Lx_max_fwd = 0.75 * W
    Lx_max_aft = 0.5 * W
    Ly_max = 0.25 * W
    Lx_fwd = (Lx_max_fwd - Lx_min) * (U - U_min) / (U_max - U_min) + Lx_min
    Lx_aft = (Lx_max_aft - Lx_min) * (U - U_min) / (U_max - U_min) + Lx_min
    Ly = (Ly_max - Ly_min) * (U - U_min) / (U_max - U_min) + Ly_min
    #
    alphas = np.arange(-np.pi, np.pi, 2 * np.pi / split_num)
    if split_type == 0:
        pos_x_fwd = Lx_fwd * np.cos(alphas)
        pos_x_aft = Lx_aft * np.cos(alphas)
        pos_x = np.where((0.0 <= np.cos(alphas)), pos_x_fwd, pos_x_aft)
        pos_y = Ly * np.sin(alphas)
    elif split_type == 1:
        pos_x_fwd = (Lx_fwd * Ly) / np.sqrt(Ly**2 + Lx_fwd**2 * np.tan(alphas) ** 2)
        pos_x_aft = (Lx_aft * Ly) / np.sqrt(Ly**2 + Lx_aft**2 * np.tan(alphas) ** 2)
        pos_x = np.where((0.0 <= np.cos(alphas)), pos_x_fwd, pos_x_aft)
        pos_y_fwd = (Lx_fwd * Ly * np.tan(alphas)) / np.sqrt(
            Ly**2 + Lx_fwd**2 * np.tan(alphas) ** 2
        )
        pos_y_aft = (Lx_aft * Ly * np.tan(alphas)) / np.sqrt(
            Ly**2 + Lx_aft**2 * np.tan(alphas) ** 2
        )
        pos_y = np.where((0.0 <= np.cos(alphas)), pos_y_fwd, pos_y_aft)
    else:
        print("Error : split_type is invalid")
    points = np.concatenate([pos_x[:, np.newaxis], pos_y[:, np.newaxis]], axis=1)
    return rotation(points, eta[2]) + eta[0:2]


def ellipse_coo(eta, nu, Lpp, B, split_num=10):
    """This funcion returns the polygon that approximates a ellipse.

    Args:
        eta (numpy.ndarray): position and headding angle (x_0, y_0, psi)
        nu (numpy.ndarray): velocity and angle velocity (u, v_m, r)
        Lpp (float): ship length
        B (float): ship breadth
        split_num (int, optional): Defaults to 10.

    Returns:
        _type_: _description_
    """
    #
    alphas = np.arange(-np.pi, np.pi, 2 * np.pi / split_num)
    pos_x = 0.75 * Lpp * np.cos(alphas)
    pos_y = B * np.sin(alphas)
    points = np.concatenate([pos_x[:, np.newaxis], pos_y[:, np.newaxis]], axis=1)
    return rotation(points, eta[2]) + eta[0:2]
