# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
import parameter
import runexe
import result_read
from datetime import datetime
import traceback

# === Configuration ===
OBS_FILE = 'obs_data_aquia.csv'
LOG_FILE = 'calibration_history_aquia.csv'
OUTPUT_DAT = 'aqui_con.dat'

def log_iteration(params, mae, mse):
    """Log iteration results to CSV."""
    file_exists = os.path.exists(LOG_FILE)
    data = params.copy()
    data['MAE'] = mae
    data['MSE'] = mse
    data['RMSE'] = np.sqrt(mse) if mse < 1e9 else 1e5
    data['Timestamp'] = datetime.now().strftime("%H:%M:%S")
    df = pd.DataFrame([data])
    try:
        df.to_csv(LOG_FILE, mode='a', header=not file_exists, index=False)
    except Exception:
        pass

def run_simulation(params_dict):
    """
    Main Interface Function
    """
    print("\n--- DEBUG: Starting Simulation ---")
    try:
        # ==========================================
        # 1. Update Input Files
        # ==========================================
        print("1. Updating input files...")
        
        # --- Flow.inp ---
        flow_lines = parameter.flow_open()
        per_val = params_dict['Perm']
        flow_lines = parameter.per_para(flow_lines, per_val, per_val, per_val)
        parameter.flow_write(flow_lines)

        # --- Geochemical.inp ---
        chemical_lines = parameter.chemical_open()
        
        # A. Cation Exchange
        fix_cation = np.array([1.0])
        fix_cation_idx = np.array([0])
        cation_vars = np.array([
            params_dict['K_Na_K'], 
            params_dict['K_Na_Ca'], 
            params_dict['K_Na_Mg'], 
            params_dict['K_Na_H']
        ])
        cation_total = parameter.para_inverse_fix(fix_cation, fix_cation_idx, cation_vars)
        chemical_lines = parameter.Exchange_cation_coeff(chemical_lines, cation_total)

        # B. Init & Bound Conditions
        h_ini = 10**(-params_dict['pH_0'])
        h_bnd = 10**(-params_dict['pH_B'])
        
        inicon_vars = np.array([
            h_ini,
            params_dict['C0_Ca'], params_dict['C0_Mg'], params_dict['C0_Na'], 
            params_dict['C0_K'],  params_dict['C0_HCO3']
        ])
        bouncon_vars = np.array([
            h_bnd,
            params_dict['CB_Ca'], params_dict['CB_Mg'], params_dict['CB_Na'], 
            params_dict['CB_K'],  params_dict['CB_HCO3']
        ])
        
        fix_vals = np.array([1.0e+0, 2.7e-4, 1.018e-1, 1.0e-5, 5.0e-2])
        fix_idxs = np.array([0, 7, 8, 9, 10])
        
        inicon_total = parameter.para_inverse_fix(fix_vals, fix_idxs, inicon_vars)
        bouncon_total = parameter.para_inverse_fix(fix_vals, fix_idxs, bouncon_vars)
        
        inicon_exp = np.expand_dims(inicon_total, 0)
        bouncon_exp = np.expand_dims(bouncon_total, 0)
        ini_boun = np.concatenate((inicon_exp, bouncon_exp), 0)
        chemical_lines = parameter.Initial_boudary(chemical_lines, ini_boun)

        # C. CEC
        chemical_lines = parameter.para_CEC(params_dict['CEC'], chemical_lines)
        parameter.chemical_write(chemical_lines)

        # ==========================================
        # 2. Run Executable
        # ==========================================
        print("2. Running EXE...")
        
        if os.path.exists(OUTPUT_DAT):
            try:
                os.remove(OUTPUT_DAT)
            except:
                pass

        runexe.run_exe()
        
        if not os.path.exists(OUTPUT_DAT):
            print(f"!!! ERROR: {OUTPUT_DAT} was NOT generated. Simulation failed.")
            return None, 1e10, 1e10

        # ==========================================
        # 3. Read Results
        # ==========================================
        print("3. Reading Results...")
        
        if not os.path.exists(OBS_FILE):
            print(f"!!! ERROR: {OBS_FILE} not found.")
            return None, 1e10, 1e10
            
        df_obs = pd.read_csv(OBS_FILE)
        
        total_sq_error = 0.0
        total_count = 0
        
        comp_map = {
            'pH':   {'aqui': 'pH',     'site': ['PH_D', 'PH'],      'scale': 1.0},
            'Ca':   {'aqui': 't_ca',   'site': ['Ca_D', 'Ca'],      'scale': 1000.0},
            'Mg':   {'aqui': 't_mg',   'site': ['Mg_D', 'Mg'],      'scale': 1000.0},
            'Na':   {'aqui': 't_na',   'site': ['Na_D', 'Na'],      'scale': 1000.0},
            'K':    {'aqui': 't_k',    'site': ['K_D', 'K'],        'scale': 1000.0},
            'HCO3': {'aqui': 't_hco3', 'site': ['HCO3_D', 'HCO3'], 'scale': 1000.0}
        }

        for comp_name, config in comp_map.items():
            obs_subset = df_obs[df_obs['Component'] == comp_name]
            if obs_subset.empty: 
                continue
            
            obs_vals = obs_subset['Value'].values
            
            # Direct read without inner try-except to avoid indentation errors
            sim_vals_raw = result_read.simulation_data(config['aqui'], config['site'])
            sim_vals = sim_vals_raw * config['scale']
            

            obs_std = np.std(obs_vals)
            

            if obs_std > 1e-9:

                weight = 1.0 / obs_std
            else:

                obs_mean = np.mean(np.abs(obs_vals))
                if obs_mean > 1e-9:
                    weight = 1.0 / obs_mean
                else:
                    weight = 1.0 
            
            min_len = min(len(sim_vals), len(obs_vals))
            s_val = sim_vals[:min_len]
            o_val = obs_vals[:min_len]
            
            valid_mask = (~np.isnan(o_val)) & (~np.isnan(s_val))
            
            if np.sum(valid_mask) > 0:
                s_clean = s_val[valid_mask]
                o_clean = o_val[valid_mask]
                
                diff = s_clean - o_clean

                weighted_diff = diff * weight
                
                total_sq_error += np.sum(weighted_diff**2)
                total_count += len(s_clean)
            
        if total_count == 0:
            print("!!! ERROR: Total count is 0. No data matched.")
            return None, 1e10, 1e10

        mse = total_sq_error / total_count
        mae = np.sqrt(mse)
        rmse = np.sqrt(mse)
        
        print(f"--- SUCCESS: Weighted_RMSE={rmse:.4e} ---")
        log_iteration(params_dict, mae, mse)
        
        return None, mse, mae

    except Exception as e:
        print(f"!!! CRITICAL EXCEPTION: {e}")
        traceback.print_exc()
        return None, 1e10, 1e10
