import os
import numpy as np

from ..core import ShipCore
from ..utils.response_models import (
    linear_delay_ode_rhs,
    first_order_delay_ode_rhs,
)
from ..utils.hull_shape import ship_coo

# try:
#     from .f2py_mmg.ode_rhs import EssoOsakaMMG
# except ImportError:
#     from .py_mmg.ode_rhs import EssoOsakaMMG
from .f2py_mmg.ode_rhs import EssoOsakaMMG

def deg2rad(deg):
    return deg * np.pi / 180


class EssoOsaka_xu(ShipCore):
    def __init__(self):
        super().__init__()
        # principle particular
        self.L = 3.0
        self.B = 0.48925
        self.dynamic_model = EssoOsakaMMG()

        # state
        self.STATE_NAME = [
            "x_position_mid [m]",
            "u_velo [m/s]",
            "y_position_mid [m]",
            "vm_velo [m/s]",
            "psi [rad]",
            "r_angvelo [rad/s]",
            "delta_rudder [rad]",
            "n_prop [rps]",
        ]
        self.STATE_UPPER_BOUND = [
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            deg2rad(35),
            20,
        ]
        self.STATE_LOWER_BOUND = [
            -np.inf,
            -np.inf,
            -np.inf,
            -np.inf,
            -np.inf,
            -np.inf,
            -deg2rad(35),
            -20,
        ]
        self.STATE_DIM = len(self.STATE_NAME)
        # action
        self.ACTION_NAME = [
            "delta_rudder_cmd [rad]",
            "n_prop_cmd [rps]",
            "true_wind_speed [m/s]",
            "true_wind_direction [rad]",
        ]
        self.ACTION_UPPER_BOUND = [
            deg2rad(35),
            20,
            np.inf,
            2 * np.pi,
        ]
        self.ACTION_LOWER_BOUND = [
            -deg2rad(35),
            -20,
            0.0,
            0.0,
        ]
        self.ACTION_DIM = len(self.ACTION_NAME)
        # observation
        self.OBSERVATION_NAME = [
            "x_position_mid_hat [m]",
            "u_velo_hat [m/s]",
            "y_position_mid_hat [m]",
            "vm_velo_hat [m/s]",
            "psi_hat [rad]",
            "r_angvelo_hat [rad/s]",
            "delta_rudder_hat [rad]",
            "n_prop_hat [rps]",
        ]
        self.OBSERVATION_UPPER_BOUND = self.STATE_UPPER_BOUND
        self.OBSERVATION_LOWER_BOUND = self.STATE_LOWER_BOUND
        self.OBSERVATION_SCALE = [
            0.03,
            0.01,
            0.03,
            0.01,
            deg2rad(0.1),
            deg2rad(0.1),
            0.0,
            0.0,
        ]
        self.OBSERVATION_DIM = len(self.OBSERVATION_NAME)

    # @profile
    def ode_rhs(self, state, action, update_params):
        x = state[0:6]
        u = [state[6], state[7], 0.0, 0.0]
        u_cmd = action[0:2]
        w = action[2:4]
        #
        derivative_state = np.empty_like(state)
        derivative_state[0:6] = self.dynamic_model.ode_rhs(x, u, w, update_params)
        # derivative_state[6] = linear_delay_ode_rhs(u[0], u_cmd[0], K=deg2rad(20))
        derivative_state[6] = linear_delay_ode_rhs(u[0], u_cmd[0], K=20)
        derivative_state[7] = linear_delay_ode_rhs(u[1], u_cmd[1], K=20)
        return derivative_state

    def observe_state(self, state, np_random=None):
        if np_random is None:
            np_random = np.random
        #
        additive_noise = np_random.normal(
            loc=np.zeros_like(self.OBSERVATION_SCALE),
            scale=self.OBSERVATION_SCALE,
        )
        observation = state + additive_noise
        return observation

    def ship_polygon(self, state):
        eta = state[[0, 2, 4]]
        polygon = ship_coo(eta, self.L, self.B)
        return polygon
