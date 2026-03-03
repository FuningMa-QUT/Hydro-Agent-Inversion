# -*- coding: utf-8 -*-
from autogen import AssistantAgent, UserProxyAgent
from llms import llm_config
import sys
import os

# === 1. Smart Logger (Prevents freezing & clean logs) ===
class SmartLogger(object):
    def __init__(self, filename='aquia_agent_log.txt'):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        # Print to console and flush immediately to avoid lag
        self.terminal.write(message)
        self.terminal.flush() 

        # Filter: Only save important progress lines to file
        keywords = [
            "Eval", "RMSE", "RESULT", "Callback", 
            "Perm", "pH", "CEC", "Error", "Traceback"
        ]
        if any(k in message for k in keywords):
            self.log.write(message)
            self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = SmartLogger()
sys.stderr = sys.stdout

# === 2. System Prompt ===
coding_orchestrator = """You are a Senior Hydrogeologist and Python Optimization Expert.

### TASK
Calibrate the Aquia Aquifer Reactive Transport Model (18 Parameters).
Minimize the Root Mean Square Error (RMSE) between simulated and observed data.

### PARAMETERS (18 Unknowns)
The optimizer provides a list `x`. Map them strictly to this dictionary:
1.  'Perm'    : x[0]  Range: [1e-12, 9.9e-12]
2.  'K_Na_K'  : x[1]  Range: [7e-3, 5e-1]
3.  'K_Na_Ca' : x[2]  Range: [1e-1, 9e-1]
4.  'K_Na_Mg' : x[3]  Range: [1e-1, 9e-1]
5.  'K_Na_H'  : x[4]  Range: [1e-6, 8e-6]
6.  'pH_0'    : x[5]  Range: [6.5, 9.0]
7.  'C0_Ca'   : x[6]  Range: [1e-3, 9e-3]
8.  'C0_Mg'   : x[7]  Range: [7e-3, 4e-2]
9.  'C0_Na'   : x[8]  Range: [5e-2, 1e-1]
10. 'C0_K'    : x[9]  Range: [1e-3, 4e-3]
11. 'C0_HCO3' : x[10] Range: [8e-3, 4e-2]
12. 'pH_B'    : x[11] Range: [6.5, 9.0]
13. 'CB_Ca'   : x[12] Range: [8e-4, 4e-3]
14. 'CB_Mg'   : x[13] Range: [8e-7, 4e-6]
15. 'CB_Na'   : x[14] Range: [7e-5, 4e-4]
16. 'CB_K'    : x[15] Range: [2e-5, 9e-5]
17. 'CB_HCO3' : x[16] Range: [1e-3, 7e-3]
18. 'CEC'     : x[17] Range: [1.0, 6.0]

### CODING REQUIREMENTS
Write a COMPLETE Python script using `scipy.optimize.differential_evolution`.

1. **INTERFACE**: 
   - `from interface_aquia import run_simulation`
   - `import numpy as np`

2. **GLOBAL COUNTER**:
   - Use a global variable `eval_count = 0` to track simulations.

3. **OBJECTIVE FUNCTION**:
   - Convert `x` to `params` dict.
   - **UNPACKING**: The interface returns `(None, mse, mae)`.
   - Code: `_, mse, _ = run_simulation(params)`
   - Calculate RMSE: `rmse = np.sqrt(mse)`
   - **PRINTING**: `print(f"Eval {eval_count}: RMSE: {rmse:.4e}", flush=True)`
   - Return `rmse`.

4. **STOPPING CRITERIA**:
   - Define a `callback(xk, convergence)`.
   - If RMSE < 0.05 (Target), return True.

5. **SETTINGS**:
   - `strategy='best1bin'`
   - `maxiter=100`
   - `popsize=6`  (Note: 18 params * 6 = 108 runs per generation. This is balanced.)
   - `disp=True`
   - `polish=True`

### FINAL OUTPUT FORMAT
Print "=== OPTIMIZATION RESULT ===" followed by all parameters and the Final RMSE.
"""

coding_executor = """
You are the Code Executor.
1. Execute the script.
2. If output contains "=== OPTIMIZATION RESULT ===", reply "TERMINATE".
3. If errors occur, show the full traceback.
"""

def run_task():
    print(f"\n========== STARTING AQUIA CALIBRATION (18 PARAMS) ==========")
    print(f">>> Logs will be saved to 'calibration_history_aquia.csv'")
    print(f">>> Note: First generation will take time (100+ simulations). Please be patient.")
    
    coder = AssistantAgent(
        name="Hydro_Coder",
        llm_config=llm_config,
        system_message=coding_orchestrator,
    )

    user = UserProxyAgent(
        name="Executor",
        system_message=coding_executor,
        is_termination_msg=lambda msg: "TERMINATE" in str(msg.get("content", "")),
        llm_config=False,
        human_input_mode="NEVER",
        code_execution_config={
            "work_dir": ".", 
            "use_docker": False,
            "last_n_messages": 1
        }
    )

    user.initiate_chat(
        coder, 
        message="Write the optimization script now. Remember to handle the 18 parameters correctly."
    )

if __name__ == "__main__":
    # Check dependencies
    if not os.path.exists("obs_data_aquia.csv"):
        print(">>> Warning: 'obs_data_aquia.csv' not found. Please run step1_process_real_obs.py first!")
    else:
        # Clean up old history for a fresh start
        if os.path.exists('calibration_history_aquia.csv'):
            try:
                os.remove('calibration_history_aquia.csv')
                print(">>> Deleted old history file.")
            except:
                pass
        
        run_task()
