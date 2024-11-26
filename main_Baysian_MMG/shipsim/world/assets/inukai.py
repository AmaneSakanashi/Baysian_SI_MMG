import os

import numpy as np

from ..core import WorldCore
from ..wind import Wind


BERTH = np.array(
    [
        [-17.367342327709544, 3.0280716041071876],
        [-17.229919542826263, 2.0116391359659676],
        [0.05456548589622058, 2.047358202565451],
        [0.1, -0.1],
        [-34.877044643377836, -0.10996771745214],
        [-34.92052111027342, 5.959692668341108],
        [0.1392873362209016, 6.1008259813208925],
        [0.07654556026854228, 2.931122867965844],
    ]
)
SURROUNDING = np.array(
    [
        [-17.4041639227645, -0.12925881818829246],
        [-6.414817004034702, -11.812384726425321],
        [14.160894167967765, -8.581250293257213],
        [15.677049039474161, -18.325461626601435],
        [37.44972032749134, -15.3465043375155],
        [45.50594945449351, -21.619280499059908],
        [80.86622975263302, -8.147556473033676],
        [80.4012760552105, -1.0939661203973283],
        [95.79745356122645, 17.217912028003163],
        [85.31361082992204, 35.80118847887229],
        [65.91636202536348, 28.995865790495714],
        [54.98601351245262, 28.87097991996618],
        [48.204300713876925, 31.123524695708458],
        [36.87042995953886, 22.787911163883823],
        [30.885356185460182, 16.475548094518018],
        [20.727043893674185, 11.136354927102106],
        [5.254496596443306, 11.688113550822017],
        [-5.74389020600862, 13.581857826197325],
        [-17.25645136976518, 13.635022885046578],
        [-17.15081445223532, 6.106747543038868],
        [-35.06010219171111, 5.951132924774719],
        [-34.61614022958422, 61.675690306113744],
        [96.6583511430393, 74.84492906869976],
        [122.51707712663774, 6.283627007806029],
        [102.43685031806156, -38.75127279270508],
        [-35.27378062838487, -32.65539042814767],
        [-35.55403543599456, -0.533620395406752],
        [-17.4041639227645, -0.12925881818829246],
    ]
)


class InukaiPond(WorldCore):
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
        self.obstacle_polygons = [BERTH, SURROUNDING]

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
