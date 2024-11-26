import numpy as np
import pandas as pd
from multiprocessing import Pool
# import ray
# ray.init(num_cpus=32)

import sys
import time

# print(sys.path)
import warnings
warnings.simplefilter('ignore')

import shipsim
import simulator
from my_src.bound import *
from ddcma import *
from my_src import *

from utils.font import font_setting

font_setting()

si = simulator.SI_obj()

ship=shipsim.EssoOsaka_xu()

# Generation mode can be selected from "random" or "stationary".
wind = shipsim.Wind(mode="stationary")

world=shipsim.OpenSea(wind=wind)

sim = shipsim.ManeuveringSimulation(
    ship=ship,
    world=world,
    dt_act = 1.0,
    dt_sim=0.1,
    solve_method="rk4", # "euler" or "rk4"train_data
    log_dir="./log/sim_data/",
    check_collide=False,
)

# # Define initial state variables––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
L = 3
B = 0.48925
### Setup for dd-CMA ###
# Setting for resart
NUM_RESTART = 8  # number of restarts with increased population size
MAX_NEVAL = 1e+7   # maximal number of f-calls
F_TARGET = 1e-6  # target function value
total_neval = 0   # total number of f-calls
iter_count = 0
### N: Dimention of CMA opt. target (maybe)
N = 62

# –--------------------------------------------------------------------------------------------------------------------------
print("N",N)

xmean0 = np.random.randn(N)
# print(len(xmean0))

lam_sig_mode = 2
if lam_sig_mode == 0:
    # 集団サイズ大，初期ステップサイズ大
    lam = 20 * N
    sigma0 = 3* np.ones(N)
elif lam_sig_mode == 1:
    # 集団サイズ大，初期ステップサイズ小
    lam = 20 * N
    sigma0 = 0.1 * np.ones(N)
elif lam_sig_mode == 2:
    # 集団サイズ小，初期ステップサイズ大
    lam = N
    sigma0 = 2 * np.ones(N)
else:
    # 集団サイズ小，初期ステップサイズ大
    lam = N
    sigma0 = 5 * np.ones(N) 

ddcma = DdCma(xmean0, sigma0, best_f=1e8)
bound = Set_bound()
LOWER_BOUND, UPPER_BOUND, FLAG_PERIODIC, period_length = bound.set_param_bound(N)

# ddcma.upper_bounding_coordinate_std(period_length)
checker = Checker(ddcma)
logger = Logger(ddcma)## ===============================================================================================-
## ===============================================================================================


# Main loop
start = time.time()
time_diff = 0
for restart in range(NUM_RESTART):        
    issatisfied = False
    fbestsofar = np.inf
    fbest_list = []
    xbest_list = []
    while not issatisfied:
        # ddcma.onestep(func=fobj)
        ddcma.onestep(func=si.fobj)
        ddcma.upper_bounding_coordinate_std(period_length)
        fbest = np.min(ddcma.arf)
        xbest = ddcma.arx[np.argmin(ddcma.arf)]

        fbest_list.append(fbest)
        xbest_list.append(xbest)

        if fbest < fbestsofar:
            fbestsofar = fbest
            xbestsofar = xbest

        ## Output parameters to csv
        opt_param = mirror(xbestsofar, LOWER_BOUND, UPPER_BOUND, FLAG_PERIODIC)

        opt_mean = pd.DataFrame(opt_param[0:31])
        opt_var = pd.DataFrame(opt_param[31:])

        opt_mean.to_csv(logger.prefix + "mean_result.csv",index=False,header=False)
        opt_var.to_csv(logger.prefix +"var_result.csv",index=False,header=False)

        # fbestsofar = min(fbest, fbestsofar)
        np.savetxt(logger.prefix + 'opt_parameters.txt', ddcma.arx)


        if fbest <= F_TARGET:
            issatisfied, condition = True, 'ftarget'
        else:
            # checker.check_iqr()
            issatisfied, condition = checker()
        # if ddcma.t % 10 == 0:
            print("===================================================")
            print("progress", ddcma.t, ddcma.neval, fbest, fbestsofar)
            logger()
    logger(condition)
    print("Terminated with condition: " + str(condition))

    # For restart
    iter_count += 1
    total_neval += ddcma.neval
    if total_neval < MAX_NEVAL and fbest > F_TARGET:
        popsize = ddcma.lam * 2
        ddcma = DdCma(xmean0, sigma0, lam=popsize, best_f=10000)
        checker = Checker(ddcma)
        logger.setcma(ddcma)
        print("Restart with popsize: " + str(ddcma.lam))
    else:
        break

# Produce a figure
# fig, axdict = logger.plot()
# for key in axdict:
#     if key not in ('xmean'):
#         axdict[key].set_yscale('log')
# plt.tight_layout()
# plt.savefig(logger.prefix + '.pdf')

print("Task Completed")