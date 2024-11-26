import numpy as np


class Wind:
    implemented_mode = ["stationary", "random"]
    f_ref = 0.5

    def __init__(self, mode="random"):
        assert mode is None or mode in self.implemented_mode
        self.mode = mode

    def reset(self, w):
        w[0] = check_plus(w[0])
        # initialize
        u, gamma = w
        self.u, self.gamma = u, rad2deg(gamma)
        if self.mode == "random":
            self.u_0, self.gamma_0 = u, rad2deg(gamma)
            self.u_bar, self.gamma_bar = u, rad2deg(gamma)
            self.t = 0.0
            self.I_Nt_u = 0.0
            self.I_Nt_gamma = 0.0
            self.u_alpha, self.u_beta = u_filter_coeff(self.u_bar, self.f_ref)
            self.gamma_alpha, self.gamma_beta = gamma_filter_coeff(self.u_bar)
        return w

    def get_state(self):
        w = np.array([self.u, deg2rad(self.gamma)])
        return w

    def step(self, dt, np_random=None):
        if self.mode == "random":
            return self._random_step(dt, np_random=np_random)
        else:
            return np.array([self.u, deg2rad(self.gamma)])

    def _random_step(self, dt, np_random=None):
        if np_random is None:
            np_random = np.random
        # get random number
        dWj_u = np.sqrt(dt) * np_random.normal()
        dWj_gamma = np.sqrt(dt) * np_random.normal()
        ### solve SDE ###
        # current step
        t = self.t
        I_Nt_u = self.I_Nt_u
        I_Nt_gamma = self.I_Nt_gamma
        # next step
        t_n = t + dt
        I_Nt_u_n = ito_integral(
            I_Nt_u,
            dt,
            dWj_u,
            self.u_alpha,
            self.u_beta,
        )
        u_n = ornstein_uhlenbeck_process(
            t_n,
            self.u_0,
            self.u_bar,
            I_Nt_u_n,
            self.u_alpha,
        )
        I_Nt_gamma_n = ito_integral(
            I_Nt_gamma,
            dt,
            dWj_gamma,
            self.gamma_alpha,
            self.gamma_beta,
        )
        gamma_n = ornstein_uhlenbeck_process(
            t_n,
            self.gamma_0,
            self.gamma_bar,
            I_Nt_gamma_n,
            self.gamma_alpha,
        )
        #
        self.u = u_n
        self.gamma = gamma_n
        self.I_Nt_u = I_Nt_u_n
        self.I_Nt_gamma = I_Nt_gamma_n
        return np.array([u_n, deg2rad(gamma_n)])


def deg2rad(deg):
    return deg * np.pi / 180


def rad2deg(rad):
    return rad * 180 / np.pi


def check_plus(var):
    if var <= 0.0:
        return 1.0e-16
    return var


def ornstein_uhlenbeck_process(t, x_0, x_bar, It, alpha):
    exp = np.exp(-alpha * t)
    x = x_0 * exp + x_bar * (1.0 - exp) + It
    return x


def ito_integral(I_Nt, dt, dWj, alpha, beta):
    sigma = np.sqrt(2.0 * np.pi) * beta
    I_Nt_n = np.exp(-alpha * dt) * (I_Nt + sigma * dWj)
    return I_Nt_n


def u_filter_coeff(u_10, f_ref):
    #   Pre-set data
    #   Davenport and Hino
    Z = 15.0
    Kfriction = 0.001  # K is used for Davenport and Hino
    alpha_hinoparam = 1.0 / 8.0
    m = 2
    #   Hino
    u_bar = np.sqrt(6.0 * Kfriction * u_10**2)
    term1 = u_10 * alpha_hinoparam / np.sqrt(Kfriction)
    term2 = (Z / 10.0) ** (2 * m * alpha_hinoparam - 1.0)
    beta_hinoparam = 1.169 * 1.0e-3 * term1 * term2
    #   Linear filter with Hino's spectrum
    #   Asymptotic value at f = 0
    Suw_H_0 = 0.2382 * u_bar**2 / beta_hinoparam
    #   Asymptotic value at f = f_ref
    Suw_H_f_ref = Suw_H_0 * (1.0 + (f_ref / beta_hinoparam) ** 2) ** (-5.0 / 6.0)
    #
    f_ref2 = f_ref**2
    pi2 = 2.0 * np.pi
    alpha2 = f_ref2 * pi2**2 * Suw_H_f_ref / (Suw_H_0 - Suw_H_f_ref)
    beta2 = Suw_H_0 * alpha2 / pi2
    #
    alpha = np.sqrt(alpha2)
    beta = np.sqrt(beta2)
    return alpha, beta


def gamma_filter_coeff(u_bar):
    sigma_theory = 2.3
    beta = sigma_theory / np.sqrt(2 * np.pi)
    alpha = (sigma_theory**2) / (2.0 * 32.0**2) * u_bar ** (3.0 / 2.0)
    return alpha, beta
