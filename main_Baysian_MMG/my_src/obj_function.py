import numpy as np
import pandas as pd
# from numba import jit

class Obj_function:
    def __init__(self) -> None:
        self.pre_disturb = pd.read_csv("inputfiles/pre-distribution_MMG_params_EssoOsaka3m.csv",header=0,index_col=0 )
        
        self.m_pre_disturb_vector = np.empty(31)
        ### Parameter Init ###
        # self.m_pre_disturb_vector[0] = self.pre_disturb.at["massx_nd", "mean"]
        # self.m_pre_disturb_vector[1] = self.pre_disturb.at["massy_nd", "mean"]
        # self.m_pre_disturb_vector[2] = self.pre_disturb.at["IzzJzz_nd", "mean"]
        # # Hull
        # self.m_pre_disturb_vector[3] = self.pre_disturb.at["xuu_nd", "mean"]
        self.m_pre_disturb_vector[0] = self.pre_disturb.at["xvr_nd", "mean"]
        self.m_pre_disturb_vector[1] = self.pre_disturb.at["yv_nd", "mean"]
        self.m_pre_disturb_vector[2] = self.pre_disturb.at["yr_nd", "mean"]
        self.m_pre_disturb_vector[3] = self.pre_disturb.at["nv_nd", "mean"]
        self.m_pre_disturb_vector[4] = self.pre_disturb.at["nr_nd", "mean"]
        self.m_pre_disturb_vector[5] = self.pre_disturb.at["coeff_drag_sway", "mean"]
        self.m_pre_disturb_vector[6] = self.pre_disturb.at["cry_cross_flow", "mean"]
        self.m_pre_disturb_vector[7] = self.pre_disturb.at["crn_cross_flow", "mean"]
        # self.m_pre_disturb_vector[12] = self.pre_disturb.at["coeff_drag_aft", "mean"]
        # Propeller
        self.m_pre_disturb_vector[8] = self.pre_disturb.at["t_prop", "mean"]
        # self.m_pre_disturb_vector[14] = self.pre_disturb.at["w_prop_zero", "mean"]
        # self.m_pre_disturb_vector[15] = self.pre_disturb.at["tau_prop", "mean"]
        # self.m_pre_disturb_vector[16] = self.pre_disturb.at["coeff_cp_prop", "mean"]
        # self.m_pre_disturb_vector[17] = self.pre_disturb.at["xp_prop_nd", "mean"]
        # self.m_pre_disturb_vector[18:21] = np.array(
        #     [
        #         self.pre_disturb.at["kt_coeff0", "mean"],
        #         self.pre_disturb.at["kt_coeff1", "mean"],
        #         self.pre_disturb.at["kt_coeff2", "mean"],
        #     ]
        # )
        self.m_pre_disturb_vector[9:17] = np.array(
            [
                self.pre_disturb.at["ai_coeff_prop0", "mean"],
                self.pre_disturb.at["ai_coeff_prop1", "mean"],
                self.pre_disturb.at["ai_coeff_prop2", "mean"],
                self.pre_disturb.at["ai_coeff_prop3", "mean"],
                self.pre_disturb.at["ai_coeff_prop4", "mean"],
                self.pre_disturb.at["ai_coeff_prop5", "mean"],
                self.pre_disturb.at["ai_coeff_prop6", "mean"],
                self.pre_disturb.at["ai_coeff_prop7", "mean"],
            ]
        )
        self.m_pre_disturb_vector[17:25] = np.array(
            [
                self.pre_disturb.at["bi_coeff_prop0", "mean"],
                self.pre_disturb.at["bi_coeff_prop1", "mean"],
                self.pre_disturb.at["bi_coeff_prop2", "mean"],
                self.pre_disturb.at["bi_coeff_prop3", "mean"],
                self.pre_disturb.at["bi_coeff_prop4", "mean"],
                self.pre_disturb.at["bi_coeff_prop5", "mean"],
                self.pre_disturb.at["bi_coeff_prop6", "mean"],
                self.pre_disturb.at["bi_coeff_prop7", "mean"],
            ]
        )
        self.m_pre_disturb_vector[25:29] = np.array(
            [
                self.pre_disturb.at["ci_coeff_prop0", "mean"],
                self.pre_disturb.at["ci_coeff_prop1", "mean"],
                self.pre_disturb.at["ci_coeff_prop2", "mean"],
                self.pre_disturb.at["ci_coeff_prop3", "mean"],
            ]
        )

        # Rudder
        # self.m_pre_disturb_vector[41] = self.pre_disturb.at["t_rudder", "mean"]
        # self.m_pre_disturb_vector[42] = self.pre_disturb.at["ah_rudder", "mean"]
        # self.m_pre_disturb_vector[43] = self.pre_disturb.at["xh_rudder_nd", "mean"]
        # self.m_pre_disturb_vector[44] = self.pre_disturb.at["kx_rudder", "mean"]
        # self.m_pre_disturb_vector[45] = self.pre_disturb.at["epsilon_rudder", "mean"]
        # self.m_pre_disturb_vector[46] = self.pre_disturb.at["lr_rudder_nd", "mean"]
        # self.m_pre_disturb_vector[47] = self.pre_disturb.at["gammaN_rudder", "mean"]
        # self.m_pre_disturb_vector[48] = self.pre_disturb.at["gammaP_rudder", "mean"]
        self.m_pre_disturb_vector[29] = self.pre_disturb.at["kx_rudder_reverse", "mean"]
        # self.m_pre_disturb_vector[50] = self.pre_disturb.at["cpr_rudder", "mean"]
        self.m_pre_disturb_vector[30] = self.pre_disturb.at["KT_bow_forward", "mean"]
        # self.m_pre_disturb_vector[52] = self.pre_disturb.at["KT_bow_reverse", "mean"]
        # self.m_pre_disturb_vector[53] = self.pre_disturb.at["aY_bow", "mean"]
        # self.m_pre_disturb_vector[54] = self.pre_disturb.at["aN_bow", "mean"]
        # self.m_pre_disturb_vector[55] = self.pre_disturb.at["KT_stern_forward", "mean"]
        # self.m_pre_disturb_vector[56] = self.pre_disturb.at["KT_stern_reverse", "mean"]
        # self.m_pre_disturb_vector[57] = self.pre_disturb.at["aY_stern", "mean"]
        # self.m_pre_disturb_vector[58] = self.pre_disturb.at["aN_stern", "mean"]

        # self.m_pre_disturb_vector[59] = self.pre_disturb.at["XX0", "mean"]
        # self.m_pre_disturb_vector[60] = self.pre_disturb.at["XX1", "mean"]
        # self.m_pre_disturb_vector[61] = self.pre_disturb.at["XX3", "mean"]
        # self.m_pre_disturb_vector[62] = self.pre_disturb.at["XX5", "mean"]
        # self.m_pre_disturb_vector[63] = self.pre_disturb.at["YY1", "mean"]
        # self.m_pre_disturb_vector[64] = self.pre_disturb.at["YY3", "mean"]
        # self.m_pre_disturb_vector[65] = self.pre_disturb.at["YY5", "mean"]
        # self.m_pre_disturb_vector[66] = self.pre_disturb.at["NN1", "mean"]
        # self.v_pre_disturb_vector[67] = self.pre_disturb.at["NN2", "mean"]
        # self.v_pre_disturb_vector[68] = self.pre_disturb.at["NN3", "mean"]

        self.v_pre_disturb_vector = np.empty(31)
        ### Parameter Init ###
        # self.v_pre_disturb_vector[0] = self.pre_disturb.at["massx_nd", "var"]
        # self.v_pre_disturb_vector[1] = self.pre_disturb.at["massy_nd", "var"]
        # self.v_pre_disturb_vector[2] = self.pre_disturb.at["IzzJzz_nd", "var"]
        # # Hull
        # self.v_pre_disturb_vector[3] = self.pre_disturb.at["xuu_nd", "var"]
        self.v_pre_disturb_vector[0] = self.pre_disturb.at["xvr_nd", "var"]
        self.v_pre_disturb_vector[1] = self.pre_disturb.at["yv_nd", "var"]
        self.v_pre_disturb_vector[2] = self.pre_disturb.at["yr_nd", "var"]
        self.v_pre_disturb_vector[3] = self.pre_disturb.at["nv_nd", "var"]
        self.v_pre_disturb_vector[4] = self.pre_disturb.at["nr_nd", "var"]
        self.v_pre_disturb_vector[5] = self.pre_disturb.at["coeff_drag_sway", "var"]
        self.v_pre_disturb_vector[6] = self.pre_disturb.at["cry_cross_flow", "var"]
        self.v_pre_disturb_vector[7] = self.pre_disturb.at["crn_cross_flow", "var"]
        # self.v_pre_disturb_vector[12] = self.pre_disturb.at["coeff_drag_aft", "var"]
        # Propeller
        self.v_pre_disturb_vector[8] = self.pre_disturb.at["t_prop", "var"]
        # self.v_pre_disturb_vector[14] = self.pre_disturb.at["w_prop_zero", "var"]
        # self.v_pre_disturb_vector[15] = self.pre_disturb.at["tau_prop", "var"]
        # self.v_pre_disturb_vector[16] = self.pre_disturb.at["coeff_cp_prop", "var"]
        # self.v_pre_disturb_vector[17] = self.pre_disturb.at["xp_prop_nd", "var"]
        # self.v_pre_disturb_vector[18:21] = np.array(
        #     [
        #         self.pre_disturb.at["kt_coeff0", "var"],
        #         self.pre_disturb.at["kt_coeff1", "var"],
        #         self.pre_disturb.at["kt_coeff2", "var"],
        #     ]
        # )
        self.v_pre_disturb_vector[9:17] = np.array(
            [
                self.pre_disturb.at["ai_coeff_prop0", "var"],
                self.pre_disturb.at["ai_coeff_prop1", "var"],
                self.pre_disturb.at["ai_coeff_prop2", "var"],
                self.pre_disturb.at["ai_coeff_prop3", "var"],
                self.pre_disturb.at["ai_coeff_prop4", "var"],
                self.pre_disturb.at["ai_coeff_prop5", "var"],
                self.pre_disturb.at["ai_coeff_prop6", "var"],
                self.pre_disturb.at["ai_coeff_prop7", "var"],
            ]
        )
        self.v_pre_disturb_vector[17:25] = np.array(
            [
                self.pre_disturb.at["bi_coeff_prop0", "var"],
                self.pre_disturb.at["bi_coeff_prop1", "var"],
                self.pre_disturb.at["bi_coeff_prop2", "var"],
                self.pre_disturb.at["bi_coeff_prop3", "var"],
                self.pre_disturb.at["bi_coeff_prop4", "var"],
                self.pre_disturb.at["bi_coeff_prop5", "var"],
                self.pre_disturb.at["bi_coeff_prop6", "var"],
                self.pre_disturb.at["bi_coeff_prop7", "var"],
            ]
        )
        self.v_pre_disturb_vector[25:29] = np.array(
            [
                self.pre_disturb.at["ci_coeff_prop0", "var"],
                self.pre_disturb.at["ci_coeff_prop1", "var"],
                self.pre_disturb.at["ci_coeff_prop2", "var"],
                self.pre_disturb.at["ci_coeff_prop3", "var"],
            ]
        )

        # Rudder
        # self.v_pre_disturb_vector[41] = self.pre_disturb.at["t_rudder", "var"]
        # self.v_pre_disturb_vector[42] = self.pre_disturb.at["ah_rudder", "var"]
        # self.v_pre_disturb_vector[43] = self.pre_disturb.at["xh_rudder_nd", "var"]
        # self.v_pre_disturb_vector[44] = self.pre_disturb.at["kx_rudder", "var"]
        # self.v_pre_disturb_vector[45] = self.pre_disturb.at["epsilon_rudder", "var"]
        # self.v_pre_disturb_vector[46] = self.pre_disturb.at["lr_rudder_nd", "var"]
        # self.v_pre_disturb_vector[47] = self.pre_disturb.at["gammaN_rudder", "var"]
        # self.v_pre_disturb_vector[48] = self.pre_disturb.at["gammaP_rudder", "var"]
        self.v_pre_disturb_vector[29] = self.pre_disturb.at["kx_rudder_reverse", "var"]
        # self.v_pre_disturb_vector[50] = self.pre_disturb.at["cpr_rudder", "var"]
        self.v_pre_disturb_vector[30] = self.pre_disturb.at["KT_bow_forward", "var"]
        # self.v_pre_disturb_vector[52] = self.pre_disturb.at["KT_bow_reverse", "var"]
        # self.v_pre_disturb_vector[53] = self.pre_disturb.at["aY_bow", "var"]
        # self.v_pre_disturb_vector[54] = self.pre_disturb.at["aN_bow", "var"]
        # self.v_pre_disturb_vector[55] = self.pre_disturb.at["KT_stern_forward", "var"]
        # self.v_pre_disturb_vector[56] = self.pre_disturb.at["KT_stern_reverse", "var"]
        # self.v_pre_disturb_vector[57] = self.pre_disturb.at["aY_stern", "var"]
        # self.v_pre_disturb_vector[58] = self.pre_disturb.at["aN_stern", "var"]

        # self.v_pre_disturb_vector[59] = self.pre_disturb.at["XX0", "var"]
        # self.v_pre_disturb_vector[60] = self.pre_disturb.at["XX1", "var"]
        # self.v_pre_disturb_vector[61] = self.pre_disturb.at["XX3", "var"]
        # self.v_pre_disturb_vector[62] = self.pre_disturb.at["XX5", "var"]
        # self.v_pre_disturb_vector[63] = self.pre_disturb.at["YY1", "var"]
        # self.v_pre_disturb_vector[64] = self.pre_disturb.at["YY3", "var"]
        # self.v_pre_disturb_vector[65] = self.pre_disturb.at["YY5", "var"]
        # self.v_pre_disturb_vector[66] = self.pre_disturb.at["NN1", "var"]
        # self.v_pre_disturb_vector[67] = self.pre_disturb.at["NN2", "var"]
        # self.v_pre_disturb_vector[68] = self.pre_disturb.at["NN3", "var"]



# @jit
    def J_3(x_sim, x_train, w_noise, w_max, w_pen,t):
        sum_error = 0
        error_array = np.zeros(6)
        for i in range(6):
            if abs(x_train[i] - x_sim[i])  > w_max or np.isnan(x_train[i] - x_sim[i]) == True :
                error_array[i] = ( w_pen / (0.1+t) ) * (w_max**2)
                sum_error += error_array[i]
            else:
                error_array[i] =  w_noise[i] * (x_train[i] - x_sim[i])**2
                sum_error += error_array[i]

        return sum_error

    def J_lkh( self, theta ):
        m_pre_distrb = self.m_pre_disturb_vector
        v_pre_distrb = np.diag(self.v_pre_disturb_vector)        
        func =  0.5 * ((theta - m_pre_distrb).T @ v_pre_distrb @ (theta - m_pre_distrb))

        return func




