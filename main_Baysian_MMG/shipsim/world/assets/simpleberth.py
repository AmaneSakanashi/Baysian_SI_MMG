import numpy as np

from ..core import WorldCore
from ..wind import Wind


BERTH = np.array(
    [
        [0.0, 0.0],
        [-100.0, 0.0],
        [-100.0, 10.0],
        [0.0, 10.0],
    ]
)


class SimpleBerth(WorldCore):
    def __init__(self, wind=Wind()):
        self.wind = wind
        # state
        self.STATE_NAME = [
            "true_wind_speed [m/s]",
            "true_wind_direction [rad]",
        ]
        self.STATE_DIM = len(self.STATE_NAME)
        self.STATE_UPPER_BOUND = [np.inf, 2 * np.pi]
        self.STATE_LOWER_BOUND = [0.0, 0.0]
        # observation
        self.OBSERVATION_NAME = [
            "true_wind_speed_hat [m/s]",
            "true_wind_direction_hat [rad]",
        ]
        self.OBSERVATION_DIM = len(self.OBSERVATION_NAME)
        self.OBSERVATION_UPPER_BOUND = self.STATE_UPPER_BOUND
        self.OBSERVATION_LOWER_BOUND = self.STATE_LOWER_BOUND
        self.OBSERVATION_SCALE = [0.0, 0.0]
        # obstacles
        self.obstacle_polygons = [BERTH]

    def reset(self, state):
        w = state
        self.wind.reset(w)
        return state

    def step(self, dt, np_random=None):
        if np_random is None:
            np_random = np.random
        w_n = self.wind.step(dt, np_random=np_random)
        state_n = w_n
        return state_n

    def get_state(self):
        w = self.wind.get_state()
        state = w
        return state

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
