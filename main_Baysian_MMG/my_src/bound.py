import numpy as np
import pandas as pd



class Set_bound:    
    def __init__(self) -> None :
        self.mean_dir = pd.read_csv("inputfiles/MMG_params_EssoOsaka3m_mean_lim.csv",header=0, index_col=0 )
        self.var_dir = pd.read_csv("inputfiles/MMG_params_EssoOsaka3m_var_lim.csv",header=0, index_col=0 )

    def read_upperlimit(self,N, dir ):
        parameter_mean_limit = dir
        mmg_params_mean_upperlim_vector = np.empty(N)
                ### Parameter Init ###
        # mmg_params_mean_upperlim_vector[0] = parameter_mean_limit.at["massx_nd", "upperlim"]
        # mmg_params_mean_upperlim_vector[1] = parameter_mean_limit.at["massy_nd", "upperlim"]
        # mmg_params_mean_upperlim_vector[2] = parameter_mean_limit.at["IzzJzz_nd", "upperlim"]
        # # Hull
        # mmg_params_mean_upperlim_vector[3] = parameter_mean_limit.at["xuu_nd", "upperlim"]
        mmg_params_mean_upperlim_vector[0] = parameter_mean_limit.at["xvr_nd", "upperlim"]
        mmg_params_mean_upperlim_vector[1] = parameter_mean_limit.at["yv_nd", "upperlim"]
        mmg_params_mean_upperlim_vector[2] = parameter_mean_limit.at["yr_nd", "upperlim"]
        mmg_params_mean_upperlim_vector[3] = parameter_mean_limit.at["nv_nd", "upperlim"]
        mmg_params_mean_upperlim_vector[4] = parameter_mean_limit.at["nr_nd", "upperlim"]
        mmg_params_mean_upperlim_vector[5] = parameter_mean_limit.at["coeff_drag_sway", "upperlim"]
        mmg_params_mean_upperlim_vector[6] = parameter_mean_limit.at["cry_cross_flow", "upperlim"]
        mmg_params_mean_upperlim_vector[7] = parameter_mean_limit.at["crn_cross_flow", "upperlim"]
        # mmg_params_mean_upperlim_vector[12] = parameter_mean_limit.at["coeff_drag_aft", "upperlim"]
        # Propeller
        mmg_params_mean_upperlim_vector[8] = parameter_mean_limit.at["t_prop", "upperlim"]
        # mmg_params_mean_upperlim_vector[14] = parameter_mean_limit.at["w_prop_zero", "upperlim"]
        # mmg_params_mean_upperlim_vector[15] = parameter_mean_limit.at["tau_prop", "upperlim"]
        # mmg_params_mean_upperlim_vector[16] = parameter_mean_limit.at["coeff_cp_prop", "upperlim"]
        # mmg_params_mean_upperlim_vector[17] = parameter_mean_limit.at["xp_prop_nd", "upperlim"]
        # mmg_params_mean_upperlim_vector[18:21] = np.array(
        #     [
        #         parameter_mean_limit.at["kt_coeff0", "upperlim"],
        #         parameter_mean_limit.at["kt_coeff1", "upperlim"],
        #         parameter_mean_limit.at["kt_coeff2", "upperlim"],
        #     ]
        # )
        mmg_params_mean_upperlim_vector[9:17] = np.array(
            [
                parameter_mean_limit.at["ai_coeff_prop0", "upperlim"],
                parameter_mean_limit.at["ai_coeff_prop1", "upperlim"],
                parameter_mean_limit.at["ai_coeff_prop2", "upperlim"],
                parameter_mean_limit.at["ai_coeff_prop3", "upperlim"],
                parameter_mean_limit.at["ai_coeff_prop4", "upperlim"],
                parameter_mean_limit.at["ai_coeff_prop5", "upperlim"],
                parameter_mean_limit.at["ai_coeff_prop6", "upperlim"],
                parameter_mean_limit.at["ai_coeff_prop7", "upperlim"],
            ]
        )
        mmg_params_mean_upperlim_vector[17:25] = np.array(
            [
                parameter_mean_limit.at["bi_coeff_prop0", "upperlim"],
                parameter_mean_limit.at["bi_coeff_prop1", "upperlim"],
                parameter_mean_limit.at["bi_coeff_prop2", "upperlim"],
                parameter_mean_limit.at["bi_coeff_prop3", "upperlim"],
                parameter_mean_limit.at["bi_coeff_prop4", "upperlim"],
                parameter_mean_limit.at["bi_coeff_prop5", "upperlim"],
                parameter_mean_limit.at["bi_coeff_prop6", "upperlim"],
                parameter_mean_limit.at["bi_coeff_prop7", "upperlim"],
            ]
        )
        mmg_params_mean_upperlim_vector[25:29] = np.array(
            [
                parameter_mean_limit.at["ci_coeff_prop0", "upperlim"],
                parameter_mean_limit.at["ci_coeff_prop1", "upperlim"],
                parameter_mean_limit.at["ci_coeff_prop2", "upperlim"],
                parameter_mean_limit.at["ci_coeff_prop3", "upperlim"],
            ]
        )

        # Rudder
        # mmg_params_mean_upperlim_vector[41] = parameter_mean_limit.at["t_rudder", "upperlim"]
        # mmg_params_mean_upperlim_vector[42] = parameter_mean_limit.at["ah_rudder", "upperlim"]
        # mmg_params_mean_upperlim_vector[43] = parameter_mean_limit.at["xh_rudder_nd", "upperlim"]
        # mmg_params_mean_upperlim_vector[44] = parameter_mean_limit.at["kx_rudder", "upperlim"]
        # mmg_params_mean_upperlim_vector[45] = parameter_mean_limit.at["epsilon_rudder", "upperlim"]
        # mmg_params_mean_upperlim_vector[46] = parameter_mean_limit.at["lr_rudder_nd", "upperlim"]
        # mmg_params_mean_upperlim_vector[47] = parameter_mean_limit.at["gammaN_rudder", "upperlim"]
        # mmg_params_mean_upperlim_vector[48] = parameter_mean_limit.at["gammaP_rudder", "upperlim"]
        mmg_params_mean_upperlim_vector[29] = parameter_mean_limit.at["kx_rudder_reverse", "upperlim"]
        # mmg_params_mean_upperlim_vector[50] = parameter_mean_limit.at["cpr_rudder", "upperlim"]
        mmg_params_mean_upperlim_vector[30] = parameter_mean_limit.at["KT_bow_forward", "upperlim"]
        # mmg_params_mean_upperlim_vector[52] = parameter_mean_limit.at["KT_bow_reverse", "upperlim"]
        # mmg_params_mean_upperlim_vector[53] = parameter_mean_limit.at["aY_bow", "upperlim"]
        # mmg_params_mean_upperlim_vector[54] = parameter_mean_limit.at["aN_bow", "upperlim"]
        # mmg_params_mean_upperlim_vector[55] = parameter_mean_limit.at["KT_stern_forward", "upperlim"]
        # mmg_params_mean_upperlim_vector[56] = parameter_mean_limit.at["KT_stern_reverse", "upperlim"]
        # mmg_params_mean_upperlim_vector[57] = parameter_mean_limit.at["aY_stern", "upperlim"]
        # mmg_params_mean_upperlim_vector[58] = parameter_mean_limit.at["aN_stern", "upperlim"]

        # mmg_params_mean_upperlim_vector[59] = parameter_mean_limit.at["XX0", "upperlim"]
        # mmg_params_mean_upperlim_vector[60] = parameter_mean_limit.at["XX1", "upperlim"]
        # mmg_params_mean_upperlim_vector[61] = parameter_mean_limit.at["XX3", "upperlim"]
        # mmg_params_mean_upperlim_vector[62] = parameter_mean_limit.at["XX5", "upperlim"]
        # mmg_params_mean_upperlim_vector[63] = parameter_mean_limit.at["YY1", "upperlim"]
        # mmg_params_mean_upperlim_vector[64] = parameter_mean_limit.at["YY3", "upperlim"]
        # mmg_params_mean_upperlim_vector[65] = parameter_mean_limit.at["YY5", "upperlim"]
        # mmg_params_mean_upperlim_vector[66] = parameter_mean_limit.at["NN1", "upperlim"]
        # mmg_params_mean_upperlim_vector[67] = parameter_mean_limit.at["NN2", "upperlim"]
        # mmg_params_mean_upperlim_vector[68] = parameter_mean_limit.at["NN3", "upperlim"]

        # print(mmg_params_mean_upperlim_vector)

        return mmg_params_mean_upperlim_vector

    # @jit
    def read_lowerlimit(self,N,dir):
        parameter_mean_limit = dir
        mmg_params_mean_lowerlim_vector = np.empty(N)
                ### Parameter Init ###
        # mmg_params_mean_lowerlim_vector[0] = parameter_mean_limit.at["massx_nd", "lowerlim"]
        # mmg_params_mean_lowerlim_vector[1] = parameter_mean_limit.at["massy_nd", "lowerlim"]
        # mmg_params_mean_lowerlim_vector[2] = parameter_mean_limit.at["IzzJzz_nd", "lowerlim"]
        # # Hull
        # mmg_params_mean_lowerlim_vector[3] = parameter_mean_limit.at["xuu_nd", "lowerlim"]
        mmg_params_mean_lowerlim_vector[0] = parameter_mean_limit.at["xvr_nd", "lowerlim"]
        mmg_params_mean_lowerlim_vector[1] = parameter_mean_limit.at["yv_nd", "lowerlim"]
        mmg_params_mean_lowerlim_vector[2] = parameter_mean_limit.at["yr_nd", "lowerlim"]
        mmg_params_mean_lowerlim_vector[3] = parameter_mean_limit.at["nv_nd", "lowerlim"]
        mmg_params_mean_lowerlim_vector[4] = parameter_mean_limit.at["nr_nd", "lowerlim"]
        mmg_params_mean_lowerlim_vector[5] = parameter_mean_limit.at["coeff_drag_sway", "lowerlim"]
        mmg_params_mean_lowerlim_vector[6] = parameter_mean_limit.at["cry_cross_flow", "lowerlim"]
        mmg_params_mean_lowerlim_vector[7] = parameter_mean_limit.at["crn_cross_flow", "lowerlim"]
        # mmg_params_mean_lowerlim_vector[12] = parameter_mean_limit.at["coeff_drag_aft", "lowerlim"]
        # Propeller
        mmg_params_mean_lowerlim_vector[8] = parameter_mean_limit.at["t_prop", "lowerlim"]
        # mmg_params_mean_lowerlim_vector[14] = parameter_mean_limit.at["w_prop_zero", "lowerlim"]
        # mmg_params_mean_lowerlim_vector[15] = parameter_mean_limit.at["tau_prop", "lowerlim"]
        # mmg_params_mean_lowerlim_vector[16] = parameter_mean_limit.at["coeff_cp_prop", "lowerlim"]
        # mmg_params_mean_lowerlim_vector[17] = parameter_mean_limit.at["xp_prop_nd", "lowerlim"]
        # mmg_params_mean_lowerlim_vector[18:21] = np.array(
        #     [
        #         parameter_mean_limit.at["kt_coeff0", "lowerlim"],
        #         parameter_mean_limit.at["kt_coeff1", "lowerlim"],
        #         parameter_mean_limit.at["kt_coeff2", "lowerlim"],
        #     ]
        # )
        mmg_params_mean_lowerlim_vector[9:17] = np.array(
            [
                parameter_mean_limit.at["ai_coeff_prop0", "lowerlim"],
                parameter_mean_limit.at["ai_coeff_prop1", "lowerlim"],
                parameter_mean_limit.at["ai_coeff_prop2", "lowerlim"],
                parameter_mean_limit.at["ai_coeff_prop3", "lowerlim"],
                parameter_mean_limit.at["ai_coeff_prop4", "lowerlim"],
                parameter_mean_limit.at["ai_coeff_prop5", "lowerlim"],
                parameter_mean_limit.at["ai_coeff_prop6", "lowerlim"],
                parameter_mean_limit.at["ai_coeff_prop7", "lowerlim"],
            ]
        )
        mmg_params_mean_lowerlim_vector[17:25] = np.array(
            [
                parameter_mean_limit.at["bi_coeff_prop0", "lowerlim"],
                parameter_mean_limit.at["bi_coeff_prop1", "lowerlim"],
                parameter_mean_limit.at["bi_coeff_prop2", "lowerlim"],
                parameter_mean_limit.at["bi_coeff_prop3", "lowerlim"],
                parameter_mean_limit.at["bi_coeff_prop4", "lowerlim"],
                parameter_mean_limit.at["bi_coeff_prop5", "lowerlim"],
                parameter_mean_limit.at["bi_coeff_prop6", "lowerlim"],
                parameter_mean_limit.at["bi_coeff_prop7", "lowerlim"],
            ]
        )
        mmg_params_mean_lowerlim_vector[25:29] = np.array(
            [
                parameter_mean_limit.at["ci_coeff_prop0", "lowerlim"],
                parameter_mean_limit.at["ci_coeff_prop1", "lowerlim"],
                parameter_mean_limit.at["ci_coeff_prop2", "lowerlim"],
                parameter_mean_limit.at["ci_coeff_prop3", "lowerlim"],
            ]
        )

        # Rudder
        # mmg_params_mean_lowerlim_vector[41] = parameter_mean_limit.at["t_rudder", "lowerlim"]
        # mmg_params_mean_lowerlim_vector[42] = parameter_mean_limit.at["ah_rudder", "lowerlim"]
        # mmg_params_mean_lowerlim_vector[43] = parameter_mean_limit.at["xh_rudder_nd", "lowerlim"]
        # mmg_params_mean_lowerlim_vector[44] = parameter_mean_limit.at["kx_rudder", "lowerlim"]
        # mmg_params_mean_lowerlim_vector[45] = parameter_mean_limit.at["epsilon_rudder", "lowerlim"]
        # mmg_params_mean_lowerlim_vector[46] = parameter_mean_limit.at["lr_rudder_nd", "lowerlim"]
        # mmg_params_mean_lowerlim_vector[47] = parameter_mean_limit.at["gammaN_rudder", "lowerlim"]
        # mmg_params_mean_lowerlim_vector[48] = parameter_mean_limit.at["gammaP_rudder", "lowerlim"]
        mmg_params_mean_lowerlim_vector[29] = parameter_mean_limit.at["kx_rudder_reverse", "lowerlim"]
        # mmg_params_mean_lowerlim_vector[50] = parameter_mean_limit.at["cpr_rudder", "lowerlim"]
        mmg_params_mean_lowerlim_vector[30] = parameter_mean_limit.at["KT_bow_forward", "lowerlim"]
        # mmg_params_mean_lowerlim_vector[52] = parameter_mean_limit.at["KT_bow_reverse", "lowerlim"]
        # mmg_params_mean_lowerlim_vector[53] = parameter_mean_limit.at["aY_bow", "lowerlim"]
        # mmg_params_mean_lowerlim_vector[54] = parameter_mean_limit.at["aN_bow", "lowerlim"]
        # mmg_params_mean_lowerlim_vector[55] = parameter_mean_limit.at["KT_stern_forward", "lowerlim"]
        # mmg_params_mean_lowerlim_vector[56] = parameter_mean_limit.at["KT_stern_reverse", "lowerlim"]
        # mmg_params_mean_lowerlim_vector[57] = parameter_mean_limit.at["aY_stern", "lowerlim"]
        # mmg_params_mean_lowerlim_vector[58] = parameter_mean_limit.at["aN_stern", "lowerlim"]

        # mmg_params_mean_lowerlim_vector[59] = parameter_mean_limit.at["XX0", "lowerlim"]
        # mmg_params_mean_lowerlim_vector[60] = parameter_mean_limit.at["XX1", "lowerlim"]
        # mmg_params_mean_lowerlim_vector[61] = parameter_mean_limit.at["XX3", "lowerlim"]
        # mmg_params_mean_lowerlim_vector[62] = parameter_mean_limit.at["XX5", "lowerlim"]
        # mmg_params_mean_lowerlim_vector[63] = parameter_mean_limit.at["YY1", "lowerlim"]
        # mmg_params_mean_lowerlim_vector[64] = parameter_mean_limit.at["YY3", "lowerlim"]
        # mmg_params_mean_lowerlim_vector[65] = parameter_mean_limit.at["YY5", "lowerlim"]
        # mmg_params_mean_lowerlim_vector[66] = parameter_mean_limit.at["NN1", "lowerlim"]
        # mmg_params_mean_lowerlim_vector[67] = parameter_mean_limit.at["NN2", "lowerlim"]
        # mmg_params_mean_lowerlim_vector[68] = parameter_mean_limit.at["NN3", "lowerlim"]

        # print(mmg_params_mean_lowerlim_vector)

        return mmg_params_mean_lowerlim_vector

        # @jit
    def set_param_bound( self, Dim ):
        m_LOWER_BOUND = self.read_lowerlimit(int(0.5 * Dim), self.mean_dir)
        m_UPPER_BOUND = self.read_upperlimit(int(0.5 * Dim), self.mean_dir)
        
        v_LOWER_BOUND = self.read_lowerlimit(int(0.5 * Dim), self.var_dir)
        v_UPPER_BOUND = self.read_upperlimit(int(0.5 * Dim), self.var_dir)

        LOWER_BOUND = np.concatenate((m_LOWER_BOUND, v_LOWER_BOUND))
        UPPER_BOUND = np.concatenate((m_UPPER_BOUND, v_UPPER_BOUND))
        # print("LOWER_BOUND", len(LOWER_BOUND),LOWER_BOUND)
        # print("UPPER_BOUND", len(UPPER_BOUND),UPPER_BOUND)
        ##--------------------------------
        FLAG_PERIODIC = np.asarray([False] * Dim )
        period_length = (UPPER_BOUND - LOWER_BOUND) * 2.0
        # period_length[FLAG_PERIODIC] /= 2.0
        # ----------------------------------------------------------------------
        # period_length[np.logical_not(np.isfinite(period_length))] = np.inf

        # return LOWER_BOUND, UPPER_BOUND, FLAG_PERIODIC
        return LOWER_BOUND, UPPER_BOUND, FLAG_PERIODIC, period_length