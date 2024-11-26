from subroutine import MMG
from subroutine import runge_kutta as rk
from subroutine import trajectory as traj
from subroutine import relative
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

### debug zw
# init

# read principal particulars from csv6
principal_particulars = pd.read_csv('inputfiles/principal_particulars_EssoOsaka3m.csv', header=0, index_col=0)
# read parameters from csv
parameter_init = pd.read_csv('inputfiles/MMG_params_EssoOsaka3m.csv', header=0, index_col=0)

# read model switch param form csv
switch = pd.read_csv('inputfiles/model_switch.csv',header=0, index_col=0)

# set save path
savepath = "output/"

# plot params
plt.rcParams["font.family"] = "Times New Roman"       # 使用するフォント
# plt.rcParams["font.serif"] = "Times New Roman"
plt.rcParams["font.size"] = 16              # 基本となるフォントの大きさ
plt.rcParams["mathtext.cal"] = "serif"      # TeX表記に関するフォント設定
plt.rcParams["mathtext.rm"] = "serif"       # TeX表記に関するフォント設定
plt.rcParams["mathtext.it"] = "serif:italic"# TeX表記に関するフォント設定
plt.rcParams["mathtext.bf"] = "serif:bold"  # TeX表記に関するフォント設定
plt.rcParams["mathtext.fontset"] = "cm"     # TeX表記に関するフォント設定

# read experiment data
exp_data = pd.read_csv('traindata/random_processed_Result_sim_0_06-Aug-2020_16_24_25_4.xlsx.csv', header=0, index_col=0) 
dt_sec = exp_data.index[2:3].values - exp_data.index[1:2].values
# instantiate mmg model
mmg2 = MMG.MmgModel(principal_particulars, parameter_init, switch)

lpp = mmg2.lpp
breadth = mmg2.B

# yasukawa's model
mmg2.switch_windtype = 3

# set start time step of MMG simulation
startstep = 0 
num_sample = 10
# set control variable
n_prop = exp_data.loc[exp_data.index[startstep:],'n_prop [rps]'].values
delta_rudder = exp_data.loc[exp_data.index[startstep:],'delta_rudder [rad]'].values
no_timestep=len(n_prop)
t = exp_data.index[startstep:].values
exp_x     = exp_data['x_position_mid [m]'].values
exp_u     = exp_data['u_velo [m/s]'].values
exp_y     = exp_data['y_position_mid [m]'].values
exp_v     = exp_data['vm_velo [m/s]'].values
exp_psi   = exp_data['psi_hat [rad]'].values
exp_r     = exp_data['r_angvelo [rad/s]'].values
# set initial sate value 
stateval  = np.zeros((no_timestep, 6)) #[x_pos, u, y_pos, vm, psi, r]
stateval[:,:] = np.nan  ## for graph visibility, filled with nan
stateval[0,0] = exp_x[0] #x_pos t=0
stateval[0,1] = exp_u[0] #u t=0
stateval[0,2] = exp_y[0] #y_pos t=0
stateval[0,3] = exp_v[0] #vm t=0
stateval[0,4] = exp_psi[0] #psi t=0
stateval[0,5] = exp_r[0] #r t=0

state_scale = np.array([ 1e-2,1e-5,1e-2,1e-4,1e-3,1e-4 ])
# set true wind (from experiment result, USE TURE WIND!!) 
windv  = exp_data.loc[exp_data.index[startstep:],'wind_velo_true [m/s]'].values
windd  = exp_data.loc[exp_data.index[startstep:],'wind_dir_true [rad]'].values
windv = windv[:, np.newaxis]
windd = windd[:, np.newaxis]
# print(windv.shape)
valid_yasukawa = np.empty([0, no_timestep, 6])


# compute MMG model by RK4
for n in range(num_sample):
    start = time.time()
    mmg2 = MMG.MmgModel(principal_particulars, parameter_init, switch)
    print('----SIMULATION NO.', n+1, '----')
    for i in range(no_timestep-1):
        # if (int(i % (100/dt_sec) )== 0):
        #         stateval[i:i+1,:] = [exp_x[i], exp_u[i], exp_y[i], exp_v[i], exp_psi[i], exp_r[i]]
        #         print("Time = : ", i*dt_sec, "[s]")
    # for i in range(1000-1):
        slope = rk.rk4_mmg(dt_sec, mmg2.hydroForceLs, exp_data.index[startstep+i:startstep+i+1].values, stateval[i:i+1,:], n_prop[i:i+1], 
                            delta_rudder[i:i+1], windv[i:i+1], windd[i:i+1])
        # fin_time = i
        if np.any(abs(slope/state_scale) > 1e+4):
            print("!!!!!!!!!!BURST!!!!!!!!!!")
            break
   
        stateval[i+1:i+2, :] = stateval[i:i+1, :] + slope[:] * dt_sec
        
        if (int(i % (100/dt_sec) )== 0):
            stateval[i+1:i+2, :] = [exp_x[i], exp_u[i], exp_y[i], exp_v[i], exp_psi[i], exp_r[i]]
            print("Time = : ", i*dt_sec, "[s]")
    

    # elapsed_time = time.time()-start
    # print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")    
    # save results
    valid_yasukawa = np.vstack((valid_yasukawa,[np.copy(stateval)]))
    # valid_yasukawa = np.copy(stateval[0:fin_time])


#mkplot
## plot tajectory
fig = plt.figure(figsize=(12, 12))
ax1 = fig.add_subplot(1, 1, 1)
ax1.set_ylabel("$\hat{x} [m]$")
ax1.set_xlabel("$\hat{y} [m]$")
ax1.plot(exp_y, exp_x, color="black", linewidth = 0.7, label="exp")
for n in range(num_sample):
    ax1.plot(valid_yasukawa[n, :, 2:3], valid_yasukawa[n, :, 0:1], color=plt.cm.tab20(n), linewidth = 0.7, label="sim: "+str(n+1))
# prot ship for exp
for i, ti in enumerate(t):
# for i in range(1):
    if ti % 10 == 0:
        # plot exp
        x1,x2,x3,x4,x5 = traj.ship_coo(
        np.array([exp_data.at[exp_data.index[i], 'x_position_mid [m]'], exp_data.at[exp_data.index[i], 'y_position_mid [m]'], exp_data.at[exp_data.index[i], 'psi_hat [rad]']]), lpp, breadth, scale=1)
        poly = plt.Polygon((x1[::-1],x2[::-1],x3[::-1],x4[::-1],x5[::-1]), fc="w", alpha=0.5, linewidth = 0.5, ec="black")
        ax1.add_patch(poly)

for i, ti in enumerate(t):
# for i in range(1):
    if ti % 10 == 0:        
        # plot exp wind relative
        exprwindend = np.array([exp_data.at[exp_data.index[i], 'y_position_mid [m]'], exp_data.at[exp_data.index[i], 'x_position_mid [m]']])
        exprwindstart = np.copy(exprwindend)
        exprwindstart[0,]   = exprwindstart[0] + exp_data.at[exp_data.index[i], 'wind_velo_relative_mid [m/s]'] *np.sin(exp_data.at[exp_data.index[i], 'wind_dir_relative_mid [rad]']+exp_data.at[exp_data.index[i], 'psi_hat [rad]'])
        exprwindstart[1,]   = exprwindstart[1] + exp_data.at[exp_data.index[i], 'wind_velo_relative_mid [m/s]'] *np.cos(exp_data.at[exp_data.index[i], 'wind_dir_relative_mid [rad]']+exp_data.at[exp_data.index[i], 'psi_hat [rad]'],)

        ax1.annotate(text='',xy=exprwindend,xytext=exprwindstart,arrowprops=dict(facecolor='gray', edgecolor='gray', width =1.0,headwidth=7.0,headlength=7.0,shrink=0.01))

        # # plot yasukawa's model
        for n in range(num_sample):
            yasukawax1, yasukawax2, yasukawax3, yasukawax4, yasukawax5 = traj.ship_coo(
            np.array([valid_yasukawa[n,i, 0], valid_yasukawa[n,i, 2], valid_yasukawa[n,i, 4]]), lpp, breadth, scale=1)
            yasukawapoly = plt.Polygon((yasukawax1[::-1], yasukawax2[::-1], yasukawax3[::-1], yasukawax4[::-1], yasukawax5[::-1]), fc="w", alpha=0.5, linewidth = 0.5, ec=plt.cm.tab20(n))
            ax1.add_patch(yasukawapoly)
        # ## relative wind
            # U_ship = np.sqrt(valid_yasukawa[n,i:i+1,1:2]**2+valid_yasukawa[n,i:i+1,3:4]**2)
            # beta = np.arctan2(valid_yasukawa[n,i:i+1,3:4], valid_yasukawa[n,i:i+1,1:2])
            # relativew = relative.true2Apparent(windv[i:i+1], U_ship, windd[i:i+1],beta, valid_yasukawa[n,i,4:5] )
            # rwindnorm = 1.0 * relativew[:,0:1]
            # rwindend = np.array([valid_yasukawa[n,i, 2], valid_yasukawa[n,i, 0]])
            # rwindstart = np.copy(rwindend)
            # rwindstart[0,]   = rwindstart[0] + rwindnorm *np.sin(valid_yasukawa[n,i, 4]+relativew[:,1:2])
            # rwindstart[1,]   = rwindstart[1] + rwindnorm *np.cos(valid_yasukawa[n,i, 4]+relativew[:,1:2])
            # ax1.annotate(text='',xy=rwindend,xytext=rwindstart,xycoords='data',\
            #     arrowprops=dict(facecolor=plt.cm.tab20(n), edgecolor=plt.cm.tab20(n), width =1.0,alpha=0.5,headwidth=7.0,headlength=7.0,shrink=0.01))

ax1.set_ylim(-5, 35)
ax1.set_xlim(-20, 10)
ax1.legend(loc='lower left')
ax1.set_aspect('equal')

fig2 = plt.figure(figsize=(16, 12))
ax1 = fig2.add_subplot(2, 3, 1)
ax1.set_xlabel("$t$ [s]")
ax1.set_ylabel("$u$ [m/s]")
ax1.plot(t, exp_u, color="black", linewidth = 0.7, label="exp")
# ax1.plot(t[0:fin_time], valid_yasukawa[:, 1:2], color="#0277BD", linewidth = 0.7, label="sim")

#vm
ax2 = fig2.add_subplot(2, 3, 2)
ax2.set_xlabel("$t$ [s]")
ax2.set_ylabel("$v_m$ [m/s]")
ax2.plot(t, exp_v, color="black", linewidth = 0.7, label="exp")
# ax2.plot(t[0:fin_time], valid_yasukawa[:, 3:4], color="#0277BD", linewidth = 0.7, label="sim")

#r
ax3 = fig2.add_subplot(2, 3, 3)
ax3.set_xlabel("$t$ [s]")
ax3.set_ylabel("$r$ [rad/s]")
ax3.plot(t, exp_r, color="black", linewidth = 0.7, label="exp")
# ax3.plot(t[0:fin_time], valid_yasukawa[:, 5:6], color="#0277BD", linewidth = 0.7, label="sim")

ax3.grid()
#x
ax4 = fig2.add_subplot(2, 3, 4)
ax4.set_xlabel("$t$ [s]")
ax4.set_ylabel("$\hat{x}$ [m/s]")
ax4.plot(t, exp_x, color="black", linewidth = 0.7, label="exp")
# ax4.plot(t[0:fin_time], valid_yasukawa[:, 0:1], color="#0277BD", linewidth = 0.7, label="sim")

#y
ax5 = fig2.add_subplot(2, 3, 5)
ax5.set_xlabel("$t$ [s]")
ax5.set_ylabel("$\hat{y}$ [m/s]")
ax5.plot(t, exp_y, color="black", linewidth = 0.7, label="exp")
# ax5.plot(t[0:fin_time], valid_yasukawa[:, 2:3], color="#0277BD", linewidth = 0.7, label="sim")


#psi
ax6 = fig2.add_subplot(2, 3, 6)
ax6.set_xlabel("$t$ [s]")
ax6.set_ylabel("$\psi$ [rad]")
ax6.plot(t, exp_psi, color="black", linewidth = 0.7, label="exp")
# ax6.plot(t[0:fin_time], valid_yasukawa[:, 4:5], color="#0277BD", linewidth = 0.7, label="sim")

## plot params for sim
for n in range(num_sample):
    ax1.plot(t, valid_yasukawa[n,:, 1:2], color=plt.cm.tab20(n), linewidth = 0.7, label="sim: "+str(n+1))
    ax2.plot(t, valid_yasukawa[n,:, 3:4], color=plt.cm.tab20(n), linewidth = 0.7, label="sim: "+str(n+1))
    ax3.plot(t, valid_yasukawa[n,:, 5:6], color=plt.cm.tab20(n), linewidth = 0.7, label="sim: "+str(n+1))
    ax4.plot(t, valid_yasukawa[n,:, 0:1], color=plt.cm.tab20(n), linewidth = 0.7, label="sim: "+str(n+1))
    ax5.plot(t, valid_yasukawa[n,:, 2:3], color=plt.cm.tab20(n), linewidth = 0.7, label="sim: "+str(n+1))
    ax6.plot(t, valid_yasukawa[n,:, 4:5], color=plt.cm.tab20(n), linewidth = 0.7, label="sim: "+str(n+1))


fig2.tight_layout()
ax1.legend()
ax4.legend()


fig.savefig( savepath + "Multi_Trj_result.pdf")
fig2.savefig( savepath + "Multi_Param_result.pdf")


## save trj to csv
HEADER = [  
            'x_position_mid [m]','u_velo [m/s]', 
            'y_position_mid [m]','vm_velo [m/s]',
            'psi_hat [rad]','r_angvelo [rad/s]'
        ]
# log = [t] + list(sim_x) + list(sim_u) + list(sim_y) + list(sim_vm) + list(sim_psi) + list(sim_r)

# df_result = pd.DataFrame(valid_yasukawa, columns=HEADER).set_index(HEADER[0])
# df_result.to_csv( savepath + "timeseries_result.csv" )

print("Figure Saved")
plt.show()
