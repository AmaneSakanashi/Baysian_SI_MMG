import numpy as np
import pandas as pd
import glob

class Read_train:
    def __init__(self) -> None:
         pass
    def read_csv(self):
            self.csv_file = glob.glob("traindata/*.csv")
        
            timestep_list = []
            startstep = 0 
            no_file = len(self.csv_file)
            ## adopt the shortest timestep
            for temp_data in self.csv_file:
                data = pd.read_csv(temp_data,header=0, index_col=0 )
                # time
                no_timestep = len(data.loc[data.index[startstep:], 'x_position_mid [m]'].values)
                timestep_list.append(no_timestep)

            min_timestep = int(min(timestep_list))

            ## read train data from each file into a 2D-array
            self.set_action_train   = np.empty((0, min_timestep,2))
            self.set_state_train    = np.empty((0, min_timestep,6))
            self.set_wind_train     = np.empty((0,min_timestep,2))

            for temp_data in self.csv_file:
                data = pd.read_csv(temp_data,header=0, index_col=0 )
                # action (rudder, propeller)        
                action_train = np.empty((min_timestep, 2 ))
                action_train[:,0]  = data.loc[data.index[startstep:min_timestep], 'delta_rudder [rad]'].values
                action_train[:,1]  = data.loc[data.index[startstep:min_timestep], 'n_prop [rps]'].values
                # state (x,u,ym,vm,psi,r)
                state_train = np.empty((min_timestep, 6 ))
                state_train[:,0] = data.loc[data.index[startstep:min_timestep], 'x_position_mid [m]'].values
                state_train[:,1] = data.loc[data.index[startstep:min_timestep], 'u_velo [m/s]'].values
                state_train[:,2] = data.loc[data.index[startstep:min_timestep], 'y_position_mid [m]'].values
                state_train[:,3] = data.loc[data.index[startstep:min_timestep], 'vm_velo [m/s]'].values
                state_train[:,4] = data.loc[data.index[startstep:min_timestep], 'psi_hat [rad]'].values
                state_train[:,5] = data.loc[data.index[startstep:min_timestep], 'r_angvelo [rad/s]'].values
                # wind ( wind_velo, wind_dir )
                wind_train = np.empty((min_timestep, 2 ))
                wind_train[:,0] = data.loc[data.index[startstep:min_timestep], 'wind_velo_true [m/s]'].values
                wind_train[:,1] = data.loc[data.index[startstep:min_timestep], 'wind_dir_true [rad]'].values

                self.set_action_train   = np.vstack((self.set_action_train,[action_train]))
                self.set_state_train    = np.vstack((self.set_state_train, [state_train]))
                self.set_wind_train     = np.vstack((self.set_wind_train, [wind_train]))

            return no_file, min_timestep, self.set_action_train, self.set_state_train, self.set_wind_train



        




