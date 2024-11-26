import numpy as np
import pandas as pd
import os
# from numba import jit

# import os
# import sys
# sys.path.append(os.getcwd())

def cma_log2csv(data, log_dir, filename):
    # log_dir = "development-of-berthing-indicator-for-trajectory-planning-develop/my_src/log/"
    log_dir = log_dir
    df = pd.DataFrame(data)
    # number = 0
    # while os.path.exists(log_dir + filename + f"_{number}.csv"):
    #     number += 1
        
    # df.to_csv(log_dir + filename + f"_{number}.csv")
    df.to_csv(log_dir + filename + ".csv")
    

if __name__ == "__main__":
    test_data = [1,2,3,6,8,9]
    filename = "test_data"
    cma_log2csv(test_data, filename)