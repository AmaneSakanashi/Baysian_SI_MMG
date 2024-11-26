import numpy as np
import matplotlib.pyplot as plt

from .ship import EssoOsaka_xu
from .world import OpenSea
from .logger import Logger
from .viewer import Viewer
from .utils import polarTrue2polarApparent as pT2pA


class ManeuveringSimulation:
    def __init__(
        self,
        ship=EssoOsaka_xu(),
        world=OpenSea(),
        dt_act=1.0,
        dt_sim=0.1,
        solve_method="euler",
        log_dir="./",
        check_collide=False,
    ):
        """Maneuvering Simulator module

        Args:
            ship (ShipCore, optional): instance of ship module. Defaults to EssoOsaka().
            world (WorldCore, optional): instance of world module. Defaults to OpenSea().
            dt_act (float, optional): Timestep of action. Defaults to 1.0.
            dt_sim (float, optional): Timestep of numerical integration. Defaults to 0.1.
            solve_method (str, optional): A method of numerical integration. Defaults to "euler".
            log_dir (str, optional): Directory of log files. Defaults to "./".
            check_collide (bool, optional): Handle whether to use the collision detection module. Defaults to False.
        """
        self.ship = ship
        self.world = world
        self.dt_act = dt_act
        self.dt_sim = dt_sim
        self.solve_method = solve_method
        self.check_collide = check_collide

        # log
        self.log_dir = log_dir
        self.logger = Logger(log_dir)
        self.viewer = Viewer(
            ship,
            world,
            self.logger,
            log_dir=log_dir,
        )
        # time
        self.TIME_NAME = ["t [s]"]
        # state
        self.STATE_NAME = self.ship.STATE_NAME + self.world.STATE_NAME
        self.STATE_UPPER_BOUND = (
            self.ship.STATE_UPPER_BOUND + self.world.STATE_UPPER_BOUND
        )
        self.STATE_LOWER_BOUND = (
            self.ship.STATE_LOWER_BOUND + self.world.STATE_LOWER_BOUND
        )
        self.STATE_DIM = len(self.STATE_NAME)
        self.ship_state_idx = self.get_state_id(self.ship.STATE_NAME)
        self.world_state_idx = self.get_state_id(self.world.STATE_NAME)
        # observation
        self.OBSERVATION_NAME = self.ship.OBSERVATION_NAME + self.world.OBSERVATION_NAME
        self.OBSERVATION_UPPER_BOUND = (
            self.ship.OBSERVATION_UPPER_BOUND + self.world.OBSERVATION_UPPER_BOUND
        )
        self.OBSERVATION_LOWER_BOUND = (
            self.ship.OBSERVATION_LOWER_BOUND + self.world.OBSERVATION_LOWER_BOUND
        )
        self.OBSERVATION_SCALE = (
            self.ship.OBSERVATION_SCALE + self.world.OBSERVATION_SCALE
        )
        self.OBSERVATION_DIM = len(self.OBSERVATION_NAME)
        # action
        self.ACTION_NAME = []
        self.ACTION_UPPER_BOUND = []
        self.ACTION_LOWER_BOUND = []
        for i, name in enumerate(self.ship.ACTION_NAME):
            if name not in self.world.STATE_NAME:
                self.ACTION_NAME.append(name)
                self.ACTION_UPPER_BOUND.append(self.ship.ACTION_UPPER_BOUND[i])
                self.ACTION_LOWER_BOUND.append(self.ship.ACTION_LOWER_BOUND[i])
        self.ACTION_DIM = len(self.ACTION_NAME)

    def reset(self, state, seed=None):
        """reset simulator

        Args:
            state (StateType): Initial condition of state variables
            seed (int or np.random.Generator, optional): random seed. Defaults to None.

        Returns:
            ObsType: observation variables
        """
        # np_random
        if seed is None:
            if not hasattr(self, "np_random"):
                seed = 100
                seed_seq = np.random.SeedSequence(seed)
                self.np_random = np.random.Generator(np.random.PCG64(seed_seq))
        elif type(seed) == int:
            seed_seq = np.random.SeedSequence(seed)
            self.np_random = np.random.Generator(np.random.PCG64(seed_seq))
        else:
            self.np_random = seed
        # split
        state = np.array(state)
        ship_state = state[self.ship_state_idx]
        world_state = state[self.world_state_idx]
        # initialization
        self.t = 0.0
        self.state = state
        self.world.reset(world_state)
        # observation
        ship_observation = self.ship.observe_state(
            ship_state,
            np_random=self.np_random,
        )
        world_observation = self.world.observe_state(
            world_state,
            np_random=self.np_random,
        )
        self.observation = np.concatenate([ship_observation, world_observation])
        # log
        header = (
            self.TIME_NAME
            + self.STATE_NAME
            + self.ACTION_NAME
            + self.OBSERVATION_NAME
            # + ["collision"]
        )
        self.logger.reset(header)
        return self.observation


        # @profile
    def step(self, update_params, action, wind, world_state=None, last_logging=False):
        """simulation step

        Args:
            
            last_logging (bool, optional): If you want to save last (next) state, change this to True. Defaults to False.

        Returns:
            ObsType: Observation variable for the next step
            bool: Handle to determine the end of simulation
            dict: Additional infomation
        """
        t, state, observation = self.t, self.state, self.observation

        # start loop
        ##### current ##################################################
        # split state
        # ship_state = state[self.ship_state_idx]
        # world_state = state[self.world_state_idx]
        ship_state = state[self.ship_state_idx]
        world_state = state[self.world_state_idx]

        ##### next #####################################################
        ### time ###
        t_n = t + self.dt_sim
        ### state ###
        # world
        world_state_n = wind
        # ship
        ship_action = np.concatenate([action, world_state])
        k1 = self.ship.ode_rhs(ship_state, ship_action, update_params)
        k2 = self.ship.ode_rhs(ship_state + 0.5 * k1 * self.dt_sim, ship_action, update_params)
        k3 = self.ship.ode_rhs(ship_state + 0.5 * k2 * self.dt_sim, ship_action, update_params)
        k4 = self.ship.ode_rhs(ship_state + 1.0 * k3 * self.dt_sim, ship_action, update_params)
        ds = (1.0 * k1 + 2.0 * k2 + 2.0 * k3 + 1.0 * k4) / 6.0
        ship_state_n = ship_state + self.dt_sim * ds
        # concat world and ship state
        state_n = np.concatenate([ship_state_n, world_state_n])
        ### observation ###
        # ship_observation_n = self.ship.observe_state(
        #     ship_state_n,
        #     np_random=self.np_random,
        # )
        # world_observation_n = self.world.observe_state(
        #     world_state_n,
        #     np_random=self.np_random,
        # )
        # observation_n = np.concatenate([ship_observation_n, world_observation_n])
        ### update ###
        t = t_n
        # state, observation = state_n, observation_n
        state = state_n
        ship_state, world_state = ship_state_n, world_state_n
        # logging last state
        # if last_logging:
        log = ([t] + state.tolist())
        self.logger.append(log)
        # postprocess for next step
        # self.t, self.state, self.observation = t, state, observation
        self.t, self.state = t, state

        # info = {"t": self.t, "state": self.state, "observation": self.observation}
        # print("!!!Calculation COMPLETED!!!")
        # return observation, info
        return self.t, self.state
        
        #### Original ver. ########
        
        # t, state, observation = self.t, self.state, self.observation
        # # start loop
        # # terminated = False
        # # steps = int(self.dt_act)
        # steps = int(self.dt_act / self.dt_sim )
        # # steps = int(self.dt_act / self.dt_sim + 0.5)
        # for _ in range(steps):
        #     ##### current ##################################################
        #     # split state
        #     ship_state = state[self.ship_state_idx]
        #     # if world_state is not None:
        #     #     _world_state = world_state
        #     # else:
        #     world_state = state[self.world_state_idx]
        #     # ### check collision ###
        #     # if self.check_collide:
        #     #     collide = self.world.check_collision(self.ship.ship_polygon(ship_state))
        #     #     if collide:
        #     #         terminated = True
        #     # else:
        #     #     collide = False
        #     ### logging ###
        #     # log = (
        #     #     [t]
        #     #     + state.tolist()
        #     #     + action.tolist()
        #     #     + observation.tolist()
        #     #     # + [collide]
        #     # )
        #     # self.logger.append(log)
        #     ##### next #####################################################
        #     ### time ###
        #     t_n = t + self.dt_sim
        #     ### state ###
        #     # world
        #     # if world_state is not None:
        #     world_state_n = world_state
        #     # else:
        #     #     world_state_n = self.world.step(self.dt_sim, np_random=self.np_random)
        #     # ship
        #     ship_action = np.concatenate([action, world_state])
        #     k1 = self.ship.ode_rhs(ship_state, ship_action, update_params)
        #     k2 = self.ship.ode_rhs(ship_state + 0.5 * k1 * self.dt_sim, ship_action, update_params)
        #     k3 = self.ship.ode_rhs(ship_state + 0.5 * k2 * self.dt_sim, ship_action, update_params)
        #     k4 = self.ship.ode_rhs(ship_state + 1.0 * k3 * self.dt_sim, ship_action, update_params)
        #     ds = (1.0 * k1 + 2.0 * k2 + 2.0 * k3 + 1.0 * k4) / 6.0
        #     ship_state_n = ship_state + self.dt_sim * ds
        #     # concat world and ship state
        #     state_n = np.concatenate([ship_state_n, world_state_n])
        #     ### observation ###
        #     ship_observation_n = self.ship.observe_state(
        #         ship_state_n,
        #         np_random=self.np_random,
        #     )
        #     world_observation_n = self.world.observe_state(
        #         world_state_n,
        #         np_random=self.np_random,
        #     )
        #     observation_n = np.concatenate([ship_observation_n, world_observation_n])
        #     ### update ###
        #     t = t_n
        #     state, observation = state_n, observation_n
        #     ship_state, world_state = ship_state_n, world_state_n
        # # logging last state
        # if last_logging:
        #     log = (
        #         [t]
        #         + state.tolist()
        #         + action.tolist()
        #         + observation.tolist()
        #         # + [collide]
        #     )
        #     self.logger.append(log)
        # # postprocess for next step
        # self.t, self.state, self.observation = t, state, observation
        # info = {"t": self.t, "state": self.state, "observation": self.observation}
        # print("!!!Calculation COMPLETED!!!")
        # return observation, terminated, info

    # @profile
    # def step(self, action, world_state=None, last_logging=False):
    #     """simulation step

    #     Args:
    #         action (ActType): Action variables
    #         last_logging (bool, optional): If you want to save last (next) state, change this to True. Defaults to False.

    #     Returns:
    #         ObsType: Observation variable for the next step
    #         bool: Handle to determine the end of simulation
    #         dict: Additional infomation
    #     """
    #     t, state, observation = self.t, self.state, self.observation
    #     # start loop
    #     terminated = False
    #     steps = int(self.dt_act / self.dt_sim + 0.5)
    #     for _ in range(steps):
    #         ##### current ##################################################
    #         # split state
    #         ship_state = state[self.ship_state_idx]
    #         if world_state is not None:
    #             _world_state = world_state
    #         else:
    #             _world_state = state[self.world_state_idx]
    #         ### check collision ###
    #         if self.check_collide:
    #             collide = self.world.check_collision(self.ship.ship_polygon(ship_state))
    #             if collide:
    #                 terminated = True
    #         else:
    #             collide = False
    #         ### logging ###
    #         log = (
    #             [t]
    #             + state.tolist()
    #             + action.tolist()
    #             + observation.tolist()
    #             + [collide]
    #         )
    #         self.logger.append(log)
    #         ##### next #####################################################
    #         ### time ###
    #         t_n = t + self.dt_sim
    #         ### state ###
    #         # world
    #         if world_state is not None:
    #             world_state_n = world_state
    #         else:
    #             world_state_n = self.world.step(self.dt_sim, np_random=self.np_random)
    #         # ship
    #         ship_action = np.concatenate([action, _world_state])
    #         if self.solve_method == "euler":
    #             ds = self.ship.ode_rhs(ship_state, ship_action)
    #             ship_state_n = ship_state + self.dt_sim * ds
    #         elif self.solve_method == "rk4":
    #             k1 = self.ship.ode_rhs(ship_state, ship_action)
    #             k2 = self.ship.ode_rhs(ship_state + 0.5 * k1 * self.dt_sim, ship_action)
    #             k3 = self.ship.ode_rhs(ship_state + 0.5 * k2 * self.dt_sim, ship_action)
    #             k4 = self.ship.ode_rhs(ship_state + 1.0 * k3 * self.dt_sim, ship_action)
    #             ds = (1.0 * k1 + 2.0 * k2 + 2.0 * k3 + 1.0 * k4) / 6.0
    #             ship_state_n = ship_state + self.dt_sim * ds
    #         else:
    #             print("solve_method not exist")
    #             break
    #         # concat world and ship state
    #         state_n = np.concatenate([ship_state_n, world_state_n])
    #         ### observation ###
    #         ship_observation_n = self.ship.observe_state(
    #             ship_state_n,
    #             np_random=self.np_random,
    #         )
    #         world_observation_n = self.world.observe_state(
    #             world_state_n,
    #             np_random=self.np_random,
    #         )
    #         observation_n = np.concatenate([ship_observation_n, world_observation_n])
    #         ### update ###
    #         t = t_n
    #         state, observation = state_n, observation_n
    #         ship_state, _world_state = ship_state_n, world_state_n
    #     # logging last state
    #     if last_logging:
    #         log = (
    #             [t]
    #             + state.tolist()
    #             + action.tolist()
    #             + observation.tolist()
    #             + [collide]
    #         )
    #         self.logger.append(log)
    #     # postprocess for next step
    #     self.t, self.state, self.observation = t, state, observation
    #     info = {"t": self.t, "state": self.state, "observation": self.observation}
    #     return observation, terminated, info

    def log2df(self):
        df = self.logger.get_df()
        # add apparent wind
        w_col = [
            "true_wind_speed [m/s]",
            "true_wind_direction [rad]",
        ]
        w_hat_col = [
            "true_wind_speed_hat [m/s]",
            "true_wind_direction_hat [rad]",
        ]
        wA_col = [
            "apparent_wind_speed [m/s]",
            "apparent_wind_direction [rad]",
        ]
        wA_hat_col = [
            "apparent_wind_speed_hat [m/s]",
            "apparent_wind_direction_hat [rad]",
        ]
        trans_col = [
            "true_wind_speed [m/s]",
            "true_wind_direction [rad]",
            "u_velo [m/s]",
            "vm_velo [m/s]",
            "psi [rad]",
        ]
        df[wA_col[0]] = df.apply(lambda x: pT2pA(*x[trans_col])[0], axis=1)
        df[wA_col[1]] = df.apply(lambda x: pT2pA(*x[trans_col])[1], axis=1)
        noise = df[w_hat_col].to_numpy() - df[w_col].to_numpy()
        df[wA_hat_col] = df[wA_col].to_numpy() + noise
        #
        return df

    def log2csv(self, fname):
        self.logger.to_csv(
            fname,
            df=self.log2df(),
        )

    def log2img(self, fname, ext_type="png"):
        self.viewer.get_x0y0_plot(fname, ext_type=ext_type)
        
        
    def log2img2(self, fname, x_emg_stop=None, ext_type="png"):
        self.viewer.get_x0y0_plot(fname, x_emg_stop, ext_type=ext_type)
        self.viewer.get_timeseries_plot(
            self.TIME_NAME[0],
            self.STATE_NAME,
            f"{fname}_state",
            ext_type=ext_type,
        )
        self.viewer.get_timeseries_plot(
            self.TIME_NAME[0],
            self.OBSERVATION_NAME,
            f"{fname}_observation",
            ext_type=ext_type,
        )
        self.viewer.get_timeseries_plot(
            self.TIME_NAME[0],
            self.ACTION_NAME,
            f"{fname}_action",
            ext_type=ext_type,
        )

    def get_t(self):
        return self.t

    def get_state_id(self, names):
        return self.get_variables_id(self.STATE_NAME, names)

    def get_observation_id(self, names):
        return self.get_variables_id(self.OBSERVATION_NAME, names)

    def get_action_id(self, names):
        return self.get_variables_id(self.ACTION_NAME, names)

    @staticmethod
    def get_variables_id(variables, names):
        ids = []
        for name in names:
            id = variables.index(name)
            ids.append(id)
        return ids
