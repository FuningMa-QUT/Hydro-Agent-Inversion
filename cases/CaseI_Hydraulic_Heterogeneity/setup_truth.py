# -*- coding: utf-8 -*-
"""
setup_truth.py

Purpose:
- Construct a synthetic "Ground Truth Model" with complex geological structures for Case I.
- Execute a steady-state groundwater flow simulation using MODFLOW-2005.
- Extract synthetic hydraulic head observations with Gaussian noise from 9 monitoring wells.
- Export the noisy observations to 'obs_data.csv' for subsequent inverse modeling.

Silent Mode:
- No detailed per-well logs are printed.
- Only a single confirmation message is displayed upon successful CSV generation.
"""

import flopy
import numpy as np
import os
import pandas as pd

# === Basic Configuration ===
workspace = os.getcwd()
model_name = 'truth_model'
exe_name = 'mf2005'  # Ensure mf2005.exe is in your PATH or workspace

# 1. Initialize the MODFLOW Model Object
ml = flopy.modflow.Modflow(model_name, exe_name=exe_name, model_ws=workspace)

# 2. Discretization: Single layer, 40x40 grid
nlay, nrow, ncol = 1, 40, 40
dis = flopy.modflow.ModflowDis(
    ml,
    nlay,
    nrow,
    ncol,
    delr=50.0,  # Grid spacing in x direction (m)
    delc=50.0,  # Grid spacing in y direction (m)
    top=50.0,
    botm=0.0
)

# 3. Boundary Conditions: Dirichlet boundaries (West-to-East gradient)
ibound = np.ones((nlay, nrow, ncol), dtype=np.int32)
strt = np.ones((nlay, nrow, ncol), dtype=np.float32) * 30.0

# West boundary: Constant head = 45 m
ibound[:, :, 0] = -1
strt[:, :, 0] = 45.0
# East boundary: Constant head = 25 m
ibound[:, :, -1] = -1
strt[:, :, -1] = 25.0

bas = flopy.modflow.ModflowBas(ml, ibound=ibound, strt=strt)

# 4. Define Complex Geological Structures (7 Distinct Zones)
hk = np.zeros((nlay, nrow, ncol), dtype=np.float32)

# Create index grids for spatial masking
y_idx, x_idx = np.indices((nrow, ncol))

# Zones 1 & 2: Background Floodplain (North/South division)
mask_north = y_idx < 20
mask_south = y_idx >= 20
hk[0, mask_north] = 2.0   # Zone 1: Northern Floodplain (Silt)
hk[0, mask_south] = 5.0   # Zone 2: Southern Floodplain (Silty Sand)

# Zones 3 & 4: Sinusoidal Paleochannel
# Function: y = 20 + 8*sin(x/6)
river_path = 20 + 8 * np.sin(x_idx / 6.0)
mask_channel = (y_idx >= river_path - 3) & (y_idx <= river_path + 3)
mask_channel_up = mask_channel & (x_idx < 20)
mask_channel_down = mask_channel & (x_idx >= 20)

hk[0, mask_channel_up] = 40.0   # Zone 3: Upstream Channel (Gravel)
hk[0, mask_channel_down] = 25.0 # Zone 4: Downstream Channel (Sand)

# Zone 5: Isolated Clay Lens (Hydraulic Barrier)
mask_clay = (y_idx > 28) & (y_idx < 35) & (x_idx > 5) & (x_idx < 15)
hk[0, mask_clay] = 0.01         # Zone 5: Clay Lens

# Zone 6: Isolated Sand Lens (Local Reservoir)
mask_sand = (y_idx > 5) & (y_idx < 12) & (x_idx > 25) & (x_idx < 35)
hk[0, mask_sand] = 35.0         # Zone 6: Sand Lens

# Zone 7: Conductive Fault Zone (Diagonal Discontinuity)
# Function: y = -x + 35 (approximate)
mask_fault = (y_idx >= -x_idx + 34) & (y_idx <= -x_idx + 36)
hk[0, mask_fault] = 80.0        # Zone 7: Conductive Fault

# Assign Hydraulic Conductivity to Layer Property Flow package
lpf = flopy.modflow.ModflowLpf(ml, hk=hk, laytyp=0)

# 5. Stress Sources: Pumping Wells
# Stress Period Data: [Layer, Row, Col, Rate (m^3/d)]
stress_period_data = {
    0: [
        [0, 20, 15, -2000.0],  # Main well located in the Paleochannel
        [0, 10, 30, -500.0]    # Secondary well located in the Sand Lens
    ]
}
wel = flopy.modflow.ModflowWel(ml, stress_period_data=stress_period_data)

# 6. Solver Settings and Output Control
pcg = flopy.modflow.ModflowPcg(ml)
oc = flopy.modflow.ModflowOc(ml, stress_period_data={(0, 0): ['save head']})

# Write MODFLOW input files and execute simulation
ml.write_input()
success, buff = ml.run_model(silent=True)

if success:
    import flopy.utils.binaryfile as bf
    hds_file = bf.HeadFile(os.path.join(workspace, model_name + '.hds'))
    head_data = hds_file.get_data(totim=1.0)
    
    # Monitoring Well Network (9 Strategic Observation Points)
    obs_locations = {
        'Obs_North_FP': (0, 5, 5),    # Northern Floodplain
        'Obs_South_FP': (0, 35, 35),  # Southern Floodplain
        'Obs_Chan_Up':  (0, 20, 10),  # Upstream Paleochannel
        'Obs_Chan_Dn':  (0, 15, 30),  # Downstream Paleochannel
        'Obs_Clay':     (0, 32, 10),  # Within Clay Lens
        'Obs_Sand':     (0, 8, 30),   # Within Sand Lens
        'Obs_Fault':    (0, 30, 5),   # Within Fault Zone
        'Obs_Mix_1':    (0, 15, 15),  # Transition Zone 1
        'Obs_Mix_2':    (0, 25, 25)   # Transition Zone 2
    }
    
    obs_names = []
    obs_heads = []
    
    # Seed for reproducibility if needed: np.random.seed(42)
    for name, idx in obs_locations.items():
        true_head = head_data[idx]
        # Add measurement noise (Gaussian: Mean=0, StdDev=0.05m)
        noise = np.random.normal(loc=0.0, scale=0.05)
        noisy_head = true_head + noise
        
        obs_names.append(name)
        obs_heads.append(noisy_head)

    # Export results to CSV for inversion stage
    output_df = pd.DataFrame({'Well_ID': obs_names, 'Hydraulic_Head': obs_heads})
    output_df.to_csv('obs_data.csv', index=False)

    # Final silent confirmation
    print("Success: 'obs_data.csv' generated with 9 noisy observations (Ground Truth Model complete).")
else:
    print("Error: MODFLOW simulation failed to generate the Truth Model.")
