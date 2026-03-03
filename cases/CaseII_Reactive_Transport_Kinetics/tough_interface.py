# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
import parameter
import run_exe
import result_read
from datetime import datetime

# === Config ===
OBS_FILE = 'obs_data_tough.csv'
LOG_FILE = 'calibration_history.csv'

def log_iteration(params, mae, mse):
    """
    Log iteration parameters and errors to CSV.
    """
    file_exists = os.path.exists(LOG_FILE)
    data = params.copy()
    data['MAE'] = mae
    data['MSE'] = mse
    data['Timestamp'] = datetime.now().strftime("%H:%M:%S")
    df = pd.DataFrame([data])
    try:
        df.to_csv(LOG_FILE, mode='a', header=not file_exists, index=False)
    except Exception:
        pass # Ignore logging errors to keep simulation running

def run_tough_simulation(params_dict):
    try:
        # 1. Prepare Parameters
        fix_cation_ecof = np.array([1.0]); fix_cation_item = np.array([0])
        cation_vars = np.array([params_dict['K_Na_K'], params_dict['K_Na_Ca']])
        
        cec_val = np.array([params_dict['CEC']])
        
        fix_inicon = np.array([1.0e+0, 1e-7, 1e-10, 1e-10]); fix_in_item = np.array([0, 1, 2, 5])
        inicon_vars = np.array([params_dict['C0_Na'], params_dict['C0_K']])
        
        fix_bouncon = np.array([1.0, 1e-7, 1.0e-10, 1.0e-10]); fix_bn_item = np.array([0, 1, 3, 4])
        bouncon_vars = np.array([params_dict['CB_Ca'], params_dict['CB_Cl']])

        # 2. Update Files
        # Flow
        flow_lines = parameter.flow_open()
        per = 1e-12 * np.ones((1,3))
        flow_lines = parameter.per_para(flow_lines, per[0,0], per[0,1], per[0,2])
        parameter.flow_write(flow_lines)
        
        # Chemical
        chemical_lines = parameter.chemical_open()
        
        cation_ecof = parameter.para_inverse_fix(fix_cation_ecof, fix_cation_item, cation_vars)
        chemical_lines = parameter.Exchange_cation_coeff(chemical_lines, cation_ecof)
        
        inicon = parameter.para_inverse_fix(fix_inicon, fix_in_item, inicon_vars)
        bouncon = parameter.para_inverse_fix(fix_bouncon, fix_bn_item, bouncon_vars)
        inicon_exp = np.expand_dims(inicon, 0); bouncon_exp = np.expand_dims(bouncon, 0)
        ini_boun = np.concatenate((inicon_exp, bouncon_exp), 0)
        chemical_lines = parameter.Initial_boudary(chemical_lines, ini_boun)
        
        chemical_lines = parameter.para_CEC(cec_val, chemical_lines)
        parameter.chemical_write(chemical_lines)
        
        # 3. Run Simulation
        if not run_exe.run_exe():
            print("Warning: TOUGHREACT execution failed or timed out.")
            return None, 9999.0, 9999.0
        
        # 4. Read Results
        item = ['t_ca+2', 't_na+', 't_k+', 't_cl-']
        item_C = ['Ca', 'Na', 'K', 'Cl']
        item_D = ['Ca_D', 'Na_D', 'K_D', 'Cl_D']
        
        sim_results = {}
        for j in range(len(item_C)):
            # Assuming inter_opt=1 performs necessary interpolation
            sim_results[item_C[j]] = result_read.simulation_data_tim(
                item=item, name_index_initem=j, column=[item_D[j], item_C[j]], inter_opt=1
            )
        
        sim_df = pd.DataFrame(sim_results)
        sim_values = sim_df.values 
        
        # 5. Calculate Error
        if not os.path.exists(OBS_FILE):
            print("Error: Observation file not found.")
            return None, 9999.0, 9999.0
            
        obs_df = pd.read_csv(OBS_FILE)
        obs_values = obs_df.values
        
        # Ensure length match
        min_len = min(len(sim_values), len(obs_values))
        if min_len == 0:
            print("Error: No valid simulation data extracted.")
            return None, 9999.0, 9999.0
            
        diff = sim_values[:min_len] - obs_values[:min_len]
        
        mae = np.mean(np.abs(diff))
        mse = np.mean(diff ** 2)  # Calculate MSE
        
        # Log to CSV
        log_iteration(params_dict, mae, mse)
        
        # Return Tuple: (Simulation Data, MSE, MAE)
        return sim_values, mse, mae

    except Exception as e:
        print(f"Interface Error: {e}")
        return None, 9999.0, 9999.0
