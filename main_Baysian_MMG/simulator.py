import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
# import ray
from shipsim.ship import EssoOsaka_xu
from shipsim.world import OpenSea

from ddcma import *
from my_src import *



class SI_trj:
    def __init__(
        self,
        ship=EssoOsaka_xu(),
        world=OpenSea(),
        dt_act=1.0,
        dt_sim=0.1,
    ):
        self.ship = ship
        self.world = world
        self.dt_act = dt_act
        self.dt_sim = dt_sim
        
        self.no_files, self.no_timestep,\
        self.set_action_train, self.set_state_train, self.set_wind_train = Read_train.read_csv(self)

        self.sim = shipsim.ManeuveringSimulation(
            ship=ship,
            world=world,
            dt_act = 1.0,
            dt_sim = 0.1,
            solve_method="rk4", # "euler" or "rk4"train_data
            log_dir="./log/sim_data/",
            check_collide=False,
        )

        N = 62
        
        L = 3
        B = 0.48925
        bound = Set_bound()   
        self.LOWER_BOUND, self.UPPER_BOUND, self.FLAG_PERIODIC, self.period_length = bound.set_param_bound(N)

        ### initial state ###
        ### w_xxx : Weight of the Obj. term
        self.w_max = 1e+2
        self.w_noise = self.sim.ship.OBSERVATION_SCALE
        self.w_pen = 1e+8


    # @ray.remote
    def trj_culc(self, x):
            w_noise, w_max, w_pen  = self.w_noise, self.w_max, self.w_pen
            update_params = x.copy()
            func = 0

            # t_list = []  
            for j in range(self.no_files):
                state_train = self.set_state_train[j,:,:]
                action_train = self.set_action_train[j,:,:]
                wind_train = self.set_wind_train[j,:,:]
                # init_ship_state     = state_train[0,:]
                # init_ship_action    = action_train[0,:]
                # init_true_wind      = wind_train[0,:]

                # init_state = np.concatenate([init_ship_state,init_ship_action,init_true_wind])

                # self.sim.reset(init_state, seed=100)

                # ---------------------------------------------------------------------------------------------------------------
                    
                # ---------------------------------------------------------------------------------------------------------------
                ### calculating trj. loop ###
                for i in range(int(self.no_timestep)):
                # for i in range(1000):    
                    if (i%(100/self.dt_sim)==0):
                        init_state = np.concatenate([state_train[i,:],action_train[i,:],wind_train[i,:]])
                        self.sim.reset(init_state, seed=100)

                    t, state_sim = self.sim.step(update_params, action_train[i], wind_train[i])
            #obj_func
                    func_i = Obj_function.J_3( state_sim, state_train[i], w_noise, w_max, w_pen,t )
                    func += (-1)*func_i
                actor = Obj_function()
                func_lkh = actor.J_lkh(update_params)
                func += (-1)* func_lkh

            return func

class SI_obj:
    def __init__(self) -> None:
            self.const_Obj = 62 * (math.log( 2 * math.pi + 1 ))

    def fobj(self, x):
            actor = SI_trj()
            params_list = mirror(x, actor.LOWER_BOUND, actor.UPPER_BOUND, actor.FLAG_PERIODIC)
            cand_results = []
            
            for set_params in params_list:
                
                ## -------------------------------------------------------

                m_params = set_params[0:31]
                v_params = set_params[31:]

                ## For degug ---------------------------------------------
                # mean_result = pd.read_csv("log/cma/mean_result.csv", header=None)
                # var_result = pd.read_csv("log/cma/var_result.csv", header=None)
                # m_params = np.array(mean_result.values.flatten())
                # v_params = np.array(var_result.values.flatten())
                ## -------------------------------------------------------
                det_v_params = abs(np.sum(v_params))

                results = []
                gen_params = np.array([np.random.normal(m, v, 3) 
                        for m, v in zip(m_params,v_params)]).T
                for i in range(gen_params.ndim):
                    # per_result_id = actor.trj_culc.remote(actor,gen_params[i]) 
                    per_result_id = actor.trj_culc(gen_params[i]) 
                    results.append(per_result_id)

                randam_params = pd.DataFrame(gen_params.T)
                randam_params.to_csv("log/cma/rand_param.csv",index=False,header=False)

                # results = ray.get(results)
                results_all = np.mean(results)   + \
                                0.5 *( math.log(det_v_params)+ self.const_Obj)
                cand_results.append(-1 * results_all)

            return  cand_results



