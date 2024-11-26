import numpy as np
import pandas as pd

class UpdateParam:
    def __init__(self):
        self.mean_result = pd.read_csv("opt_result/mean_result.csv", header=None)
        self.var_result = pd.read_csv("opt_result/var_result.csv", header=None)

        self.output_path = "log/sim_data/"
        parameter_init = pd.read_csv(
            "inputfiles/MMG_params_EssoOsaka3m.csv", header=0, index_col=0
        )
        #
        self.mmg_params_vector = np.empty(len(parameter_init))
        ### Parameter Init ###
        self.mmg_params_vector[0] = parameter_init.at["massx_nd", "value"]
        self.mmg_params_vector[1] = parameter_init.at["massy_nd", "value"]
        self.mmg_params_vector[2] = parameter_init.at["IzzJzz_nd", "value"]
        # Hull
        self.mmg_params_vector[3] = parameter_init.at["xuu_nd", "value"]
        self.mmg_params_vector[4] = parameter_init.at["xvr_nd", "value"]
        self.mmg_params_vector[5] = parameter_init.at["yv_nd", "value"]
        self.mmg_params_vector[6] = parameter_init.at["yr_nd", "value"]
        self.mmg_params_vector[7] = parameter_init.at["nv_nd", "value"]
        self.mmg_params_vector[8] = parameter_init.at["nr_nd", "value"]
        self.mmg_params_vector[9] = parameter_init.at["coeff_drag_sway", "value"]
        self.mmg_params_vector[10] = parameter_init.at["cry_cross_flow", "value"]
        self.mmg_params_vector[11] = parameter_init.at["crn_cross_flow", "value"]
        self.mmg_params_vector[12] = parameter_init.at["coeff_drag_aft", "value"]
        # Propeller
        self.mmg_params_vector[13] = parameter_init.at["t_prop", "value"]
        self.mmg_params_vector[14] = parameter_init.at["w_prop_zero", "value"]
        self.mmg_params_vector[15] = parameter_init.at["tau_prop", "value"]
        self.mmg_params_vector[16] = parameter_init.at["coeff_cp_prop", "value"]
        self.mmg_params_vector[17] = parameter_init.at["xp_prop_nd", "value"]
        self.mmg_params_vector[18:21] = np.array(
            [
                parameter_init.at["kt_coeff0", "value"],
                parameter_init.at["kt_coeff1", "value"],
                parameter_init.at["kt_coeff2", "value"],
            ]
        )
        self.mmg_params_vector[21:29] = np.array(
            [
                parameter_init.at["ai_coeff_prop0", "value"],
                parameter_init.at["ai_coeff_prop1", "value"],
                parameter_init.at["ai_coeff_prop2", "value"],
                parameter_init.at["ai_coeff_prop3", "value"],
                parameter_init.at["ai_coeff_prop4", "value"],
                parameter_init.at["ai_coeff_prop5", "value"],
                parameter_init.at["ai_coeff_prop6", "value"],
                parameter_init.at["ai_coeff_prop7", "value"],
            ]
        )
        self.mmg_params_vector[29:37] = np.array(
            [
                parameter_init.at["bi_coeff_prop0", "value"],
                parameter_init.at["bi_coeff_prop1", "value"],
                parameter_init.at["bi_coeff_prop2", "value"],
                parameter_init.at["bi_coeff_prop3", "value"],
                parameter_init.at["bi_coeff_prop4", "value"],
                parameter_init.at["bi_coeff_prop5", "value"],
                parameter_init.at["bi_coeff_prop6", "value"],
                parameter_init.at["bi_coeff_prop7", "value"],
            ]
        )
        self.mmg_params_vector[37:41] = np.array(
            [
                parameter_init.at["ci_coeff_prop0", "value"],
                parameter_init.at["ci_coeff_prop1", "value"],
                parameter_init.at["ci_coeff_prop2", "value"],
                parameter_init.at["ci_coeff_prop3", "value"],
            ]
        )

        # Rudder
        self.mmg_params_vector[41] = parameter_init.at["t_rudder", "value"]
        self.mmg_params_vector[42] = parameter_init.at["ah_rudder", "value"]
        self.mmg_params_vector[43] = parameter_init.at["xh_rudder_nd", "value"]
        self.mmg_params_vector[44] = parameter_init.at["kx_rudder", "value"]
        self.mmg_params_vector[45] = parameter_init.at["epsilon_rudder", "value"]
        self.mmg_params_vector[46] = parameter_init.at["lr_rudder_nd", "value"]
        self.mmg_params_vector[47] = parameter_init.at["gammaN_rudder", "value"]
        self.mmg_params_vector[48] = parameter_init.at["gammaP_rudder", "value"]
        self.mmg_params_vector[49] = parameter_init.at["kx_rudder_reverse", "value"]
        self.mmg_params_vector[50] = parameter_init.at["cpr_rudder", "value"]
        self.mmg_params_vector[51] = parameter_init.at["KT_bow_forward", "value"]
        self.mmg_params_vector[52] = parameter_init.at["KT_bow_reverse", "value"]
        self.mmg_params_vector[53] = parameter_init.at["aY_bow", "value"]
        self.mmg_params_vector[54] = parameter_init.at["aN_bow", "value"]
        self.mmg_params_vector[55] = parameter_init.at["KT_stern_forward", "value"]
        self.mmg_params_vector[56] = parameter_init.at["KT_stern_reverse", "value"]
        self.mmg_params_vector[57] = parameter_init.at["aY_stern", "value"]
        self.mmg_params_vector[58] = parameter_init.at["aN_stern", "value"]

        self.mmg_params_vector[59] = parameter_init.at["XX0", "value"]
        self.mmg_params_vector[60] = parameter_init.at["XX1", "value"]
        self.mmg_params_vector[61] = parameter_init.at["XX3", "value"]
        self.mmg_params_vector[62] = parameter_init.at["XX5", "value"]
        self.mmg_params_vector[63] = parameter_init.at["YY1", "value"]
        self.mmg_params_vector[64] = parameter_init.at["YY3", "value"]
        self.mmg_params_vector[65] = parameter_init.at["YY5", "value"]
        self.mmg_params_vector[66] = parameter_init.at["NN1", "value"]
        self.mmg_params_vector[67] = parameter_init.at["NN2", "value"]
        self.mmg_params_vector[68] = parameter_init.at["NN3", "value"]

    def update(self):           
            m_params = np.array(self.mean_result.values.flatten())
            v_params = np.array(self.var_result.values.flatten())  

            update_params = np.random.normal(m_params, v_params)

            update_index = np.array([4,5,6,7,8,9,10,11,
                            13,
                            22,23,24,25,26,27,28,29,
                            30,31,32,33,34,35,36,37,
                            38,39,40,41,
                            49,51])

            update_mmg_params = self.mmg_params_vector

            for i in range(31):
                update_mmg_params[update_index[i]] = update_params[i]

            return update_mmg_params
        
    def setParams(self, mmg_params_vector):
        """ update parameters (hydro derivatives etc) of MMG model
        
        Args:
            parameter(dataflaminputfiles/principal_particulars_EssoOsaka3m.csv
        """
        self.MassX_nd      = mmg_params_vector[0] 
        self.MassY_nd      = mmg_params_vector[1] 
        self.IJzz_nd       = mmg_params_vector[2]
        # Hull
        self.Xuu_nd        = mmg_params_vector[3] 
        self.Xvr_nd        = mmg_params_vector[4]
        self.Yv_nd         = mmg_params_vector[5]
        self.Yr_nd         = mmg_params_vector[6]
        self.Nv_nd         = mmg_params_vector[7] 
        self.Nr_nd         = mmg_params_vector[8]
        self.CD            = mmg_params_vector[9]#0.500
        self.C_rY          = mmg_params_vector[10] #1.00
        self.C_rN          = mmg_params_vector[11] #0.50
        self.X_0F_nd       = self.Xuu_nd
        self.X_0A_nd       = mmg_params_vector[12]
        # Propeller
        self.t_prop        = mmg_params_vector[13] 
        self.wP0           = mmg_params_vector[14]  
        self.tau           = mmg_params_vector[15]
        self.CP_nd         = mmg_params_vector[16] 
        self.xP_nd         = mmg_params_vector[17] 
        self.kt_coeff      = np.array([mmg_params_vector[18], mmg_params_vector[19], 
                                        mmg_params_vector[20]]) 
        self.Ai            = np.array([mmg_params_vector[21], mmg_params_vector[22], 
                                        mmg_params_vector[23], mmg_params_vector[24],
                                        mmg_params_vector[25], mmg_params_vector[26],
                                        mmg_params_vector[27], mmg_params_vector[28]]) 
        self.Bi            = np.array([mmg_params_vector[29], mmg_params_vector[30], 
                                        mmg_params_vector[31], mmg_params_vector[32],
                                        mmg_params_vector[33], mmg_params_vector[34],
                                       mmg_params_vector[35], mmg_params_vector[36]])
        self.Ci            = np.array([mmg_params_vector[37], mmg_params_vector[38], 
                                        mmg_params_vector[39], mmg_params_vector[40]]) 
        self.Jmin          =  -0.5 # coeff. from exp
        self.alpha_p       =  1.5  # coeff. from exp   
        
        # Rudder
        self.t_rudder           = mmg_params_vector[41] 
        self.ah_rudder          = mmg_params_vector[42]
        self.xh_rudder_nd       = mmg_params_vector[43]
        self.lr_rudder_nd       = mmg_params_vector[46]
        self.kx_rudder          = mmg_params_vector[44]
        self.kx_rudder_reverse  = mmg_params_vector[49]
        self.epsilon_rudder     = mmg_params_vector[45]
        self.cpr_rudder         = mmg_params_vector[50] 
        self.gammaN             = mmg_params_vector[47]
        self.gammaP             = mmg_params_vector[48]

        # Wind
        self.XX0 = mmg_params_vector[59] 
        self.XX1 = mmg_params_vector[60] 
        self.XX3 = mmg_params_vector[61] 
        self.XX5 = mmg_params_vector[62] 
        self.YY5 = mmg_params_vector[65]
        self.NN1 = mmg_params_vector[66]  
        self.NN2 = mmg_params_vector[67]  
        self.NN3 = mmg_params_vector[68]
        self.KK1 = 0
        self.KK2 = 0
        self.KK3 = 0
        self.KK5 = 0  
        # 