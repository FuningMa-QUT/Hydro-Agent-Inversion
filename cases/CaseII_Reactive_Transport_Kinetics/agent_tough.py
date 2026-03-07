# -*- coding: utf-8 -*-
from autogen import AssistantAgent, UserProxyAgent
from llms import llm_config
import sys
import os

class CleanLogger(object):
    def __init__(self, filename='tough_agent_log.txt'):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()

        valid_keywords = [
            "Eval",                 
            "RMSE",            
            "Callback",       
            "OPTIMIZATION",     
            "RESULT",           
            "K_Na_K", "K_Na_Ca", "C0_Na", "C0_K", "CB_Ca", "CB_Cl", "CEC"
        ]
        
        if any(k in message for k in valid_keywords) and "to Executor" not in message:
            self.log.write(message)
            self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = CleanLogger()
sys.stderr = sys.stdout

coding_orchestrator = """You are a Senior Hydrogeologist and Optimization Expert.

### TASK
Calibrate TOUGHREACT parameters to minimize the Root Mean Square Error (RMSE).
The target is high quality fit (RMSE < 5.0e-5).

### PARAMETERS
The optimizer provides a vector `x`. Map it strictly to this dictionary:
1. 'K_Na_K'    : x[0] (Range: 0.1 - 0.3)
2. 'K_Na_Ca'   : x[1] (Range: 0.25 - 0.42)
3. 'C0_Na'     : x[2] (Range: 8e-4 - 8e-3)
4. 'C0_K'      : x[3] (Range: 1e-4 - 1e-3)
5. 'CB_Ca'     : x[4] (Range: 3e-4 - 8e-4)
6. 'CB_Cl'     : x[5] (Range: 5e-4 - 5e-3)
7. 'CEC'       : x[6] (Range: 5e-3 - 5e-2)

### STRATEGY (Autonomous)
1. **Analyze the problem**: Choose the most robust optimization algorithm from `scipy.optimize` (e.g., `differential_evolution`, `dual_annealing`, or `minimize` with global wrappers).
2. **Configure Settings**: Select appropriate hyperparameters (max iterations, population size, tolerances) suitable for a 7-parameter non-linear groundwater model.
3. **Justification**: Briefly print which algorithm was selected and why at the start of the script.

### CODING REQUIREMENTS
1. **Interface**: `from tough_interface import run_tough_simulation`.
2. **Objective Function**:
   - Map NumPy array `x` to the dictionary `params`.
   - Call `_, mse, _ = run_tough_simulation(params)`.
   - Calculate `rmse = np.sqrt(mse)`.
   - Handle failed simulations (if mse is 9999) by returning a high penalty.
   - Print progress: `print(f"Eval {count}: RMSE: {rmse:.4e}")`.
3. **Termination**: Include a callback if supported to stop when RMSE < 5.0e-5.

### FINAL OUTPUT FORMAT
Print exactly:
=== OPTIMIZATION RESULT ===
Algorithm: [Selected Algorithm]
[Parameters with Values]
Final RMSE: [Value]
"""

coding_executor = """
You are the Code Executor.
1. Execute the Python script.
2. If output contains "=== OPTIMIZATION RESULT ===", reply "TERMINATE".
3. If errors occur, show the full traceback.
"""

def run_calibration_task():
    print(f"\n========== STARTING AUTONOMOUS CALIBRATION ==========")
    
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
        message="Generate and run an optimization script to find the best parameters."
    )

if __name__ == "__main__":
    if not os.path.exists("obs_data_tough.csv"):
        print("CRITICAL ERROR: 'obs_data_tough.csv' missing.")
    else:
        run_calibration_task()
