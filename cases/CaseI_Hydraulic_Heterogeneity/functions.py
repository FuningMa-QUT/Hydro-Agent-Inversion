# -*- coding: utf-8 -*-
"""
functions.py (Fixed MSE Calculation)

Updates:
- Fixed the return statement to correctly calculate and return MSE.
"""
import flopy
import numpy as np
import os
import pandas as pd

# === Global Config ===
EXE_NAME = 'mf2005.exe'
MODEL_WS = os.getcwd()
TRUTH_CSV = 'obs_data.csv'

def simulation_objective_calibration(params_dict: dict):
    """
    Accepts dictionary with 7 geological keys.
    Returns: (None, mse, mae)
    """
    model_name = 'ai_sim_model'
    
    # Clean up previous run
    if os.path.exists(model_name + '.hds'):
        try: os.remove(model_name + '.hds')
        except: pass

    # 1. Rebuild Model (40x40 Grid)
    # Note: Ensure verbose=False to keep console clean
    ml = flopy.modflow.Modflow(model_name, exe_name=EXE_NAME, model_ws=MODEL_WS, verbose=False)
    nlay, nrow, ncol = 1, 40, 40
    dis = flopy.modflow.ModflowDis(ml, nlay, nrow, ncol, delr=50, delc=50, top=50, botm=0)
    
    ibound = np.ones((nlay, nrow, ncol), dtype=np.int32)
    strt = np.ones((nlay, nrow, ncol), dtype=np.float32) * 30.0
    ibound[:, :, 0] = -1; strt[:, :, 0] = 45.0
    ibound[:, :, -1] = -1; strt[:, :, -1] = 25.0
    bas = flopy.modflow.ModflowBas(ml, ibound=ibound, strt=strt)
    
    # === 2. Map Parameters to Geology ===
    hk = np.zeros((nlay, nrow, ncol), dtype=np.float32)
    y, x = np.indices((nrow, ncol))
    
    # Retrieve params (with defaults to prevent crash)
    k_north = float(params_dict.get('K_North', 1.0))
    k_south = float(params_dict.get('K_South', 1.0))
    k_ch_up = float(params_dict.get('K_Chan_Up', 10.0))
    k_ch_dn = float(params_dict.get('K_Chan_Dn', 10.0))
    k_clay = float(params_dict.get('K_Clay', 0.01))
    k_sand = float(params_dict.get('K_Sand', 10.0))
    k_fault = float(params_dict.get('K_Fault', 10.0))

    # Apply Masks (Order matters!)
    # 1. Background
    mask_north = y < 20
    mask_south = y >= 20
    hk[0, mask_north] = k_north
    hk[0, mask_south] = k_south
    
    # 2. Channel
    river_path = 20 + 8 * np.sin(x / 6.0)
    mask_channel = (y >= river_path - 3) & (y <= river_path + 3)
    mask_channel_up = mask_channel & (x < 20)
    mask_channel_down = mask_channel & (x >= 20)
    hk[0, mask_channel_up] = k_ch_up
    hk[0, mask_channel_down] = k_ch_dn
    
    # 3. Anomalies
    mask_clay = (y > 28) & (y < 35) & (x > 5) & (x < 15)
    hk[0, mask_clay] = k_clay
    
    mask_sand = (y > 5) & (y < 12) & (x > 25) & (x < 35)
    hk[0, mask_sand] = k_sand
    
    mask_fault = (y >= -x + 34) & (y <= -x + 36)
    hk[0, mask_fault] = k_fault

    lpf = flopy.modflow.ModflowLpf(ml, hk=hk, laytyp=0)
    
    # 3. Wells
    stress_period_data = {
        0: [[0, 20, 15, -2000.0], [0, 10, 30, -500.0]]
    }
    wel = flopy.modflow.ModflowWel(ml, stress_period_data=stress_period_data)
    
    pcg = flopy.modflow.ModflowPcg(ml)
    oc = flopy.modflow.ModflowOc(ml, stress_period_data={(0, 0): ['save head']})
    ml.write_input()
    
    # Run quietly
    success, buff = ml.run_model(silent=True)
    
    # Return penalty if failed
    if not success: return None, 1e10, 1e10

    # 4. Calculate Error
    try:
        import flopy.utils.binaryfile as bf
        hds = bf.HeadFile(os.path.join(MODEL_WS, model_name + '.hds'))
        sim_head = hds.get_data(totim=1.0)
        hds.close()
        
        df_true = pd.read_csv(TRUTH_CSV)
        real_vals = df_true['Head'].values
        
        # Define observation indices (Must match setup_truth.py)
        # obs_points = { ... keys ... }
        # Assuming order in CSV matches this order of extraction
        obs_coords = [
            (0, 5, 5), (0, 35, 35), (0, 20, 10), (0, 15, 30),
            (0, 32, 10), (0, 8, 30), (0, 30, 5), (0, 15, 15), (0, 25, 25)
        ]
        
        sim_vals = []
        for idx in obs_coords:
            sim_vals.append(sim_head[idx])
        sim_vals = np.array(sim_vals)
        
        diff = sim_vals - real_vals
        
        # === 核心修改点 ===
        mae = np.mean(np.abs(diff))
        mse = np.mean(diff ** 2)  # 计算 MSE
        
        # 返回正确的 MSE，而不是 0.0
        return None, mse, mae
        
    except Exception as e:
        # print(f"Error calculating objective: {e}")
        return None, 1e10, 1e10
