import numpy as np

from ..core import WorldCore
from ..wind import Wind
from ..utils import latlon2local


berth_latlon = [135.4110527974003, 34.63582835949745]
offshore_latlon = [135.3908711887164, 34.63044010359711]
berth_point = [0.0, 0.0]
offshore_point = latlon2local(offshore_latlon, berth_latlon)

ISLAND1 = np.array(
    [
        latlon2local([135.4075659148824, 34.63331349314861], berth_latlon),
        latlon2local([135.409013421793, 34.63433838300215], berth_latlon),
        latlon2local([135.4087303297437, 34.63461327228075], berth_latlon),
        latlon2local([135.4073041771545, 34.63357567403271], berth_latlon),
    ]
)
ISLAND2 = np.array(
    [
        latlon2local([135.3832844905106, 34.62258591726036], berth_latlon),
        latlon2local([135.4012525852844, 34.61076173850785], berth_latlon),
        latlon2local([135.4170588173345, 34.61214182728062], berth_latlon),
        latlon2local([135.417485348258, 34.61075624059106], berth_latlon),
        latlon2local([135.4191141980484, 34.61082231010572], berth_latlon),
        latlon2local([135.420370538332, 34.61286937149811], berth_latlon),
        latlon2local([135.4183154183816, 34.61956966205551], berth_latlon),
        latlon2local([135.4037456048707, 34.61972919204671], berth_latlon),
        latlon2local([135.4022019158979, 34.62062341146505], berth_latlon),
        latlon2local([135.3995902710798, 34.62066662542693], berth_latlon),
        latlon2local([135.3894727043345, 34.62846779521559], berth_latlon),
        latlon2local([135.3872326223813, 34.62632692952124], berth_latlon),
    ]
)
ISLAND3 = np.array(
    [
        latlon2local([135.4122804812167, 34.63719250244569], berth_latlon),
        latlon2local([135.4148075593339, 34.63481069207768], berth_latlon),
        latlon2local([135.4074825664467, 34.62948790910406], berth_latlon),
        latlon2local([135.4103133191639, 34.6258278194304], berth_latlon),
        latlon2local([135.4205728299486, 34.62581008636131], berth_latlon),
        latlon2local([135.4214208536259, 34.62369408132048], berth_latlon),
        latlon2local([135.4221359362829, 34.62187286614306], berth_latlon),
        latlon2local([135.4325996335653, 34.62456205505559], berth_latlon),
        latlon2local([135.4418330726762, 34.62874520258693], berth_latlon),
        latlon2local([135.4392431564526, 34.6412226740734], berth_latlon),
        latlon2local([135.4323488040034, 34.64402674794113], berth_latlon),
        latlon2local([135.424268759466, 34.64264468359747], berth_latlon),
        latlon2local([135.4198785278259, 34.64385897693621], berth_latlon),
        latlon2local([135.4177990818562, 34.64446533562957], berth_latlon),
        latlon2local([135.4165271223093, 34.64481523275914], berth_latlon),
        latlon2local([135.3988265072213, 34.63851122174749], berth_latlon),
        latlon2local([135.3975567083233, 34.638993080665], berth_latlon),
        latlon2local([135.3926796438165, 34.63545207064797], berth_latlon),
        latlon2local([135.3937758169744, 34.63052292285542], berth_latlon),
        latlon2local([135.3965693463174, 34.62850807842142], berth_latlon),
        latlon2local([135.3987293060217, 34.63132486523259], berth_latlon),
        latlon2local([135.4044902466475, 34.63153524065125], berth_latlon),
    ]
)


class Nanko(WorldCore):
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
        self.obstacle_polygons = [ISLAND1, ISLAND2, ISLAND3]

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
