# -*- coding: utf-8 -*-
"""
benchmark_agent.py (Optimized for Data Logging)

Purpose:
- Use AutoGen multi-agent framework to calibrate MODFLOW.
- UPDATED: Now forces the agent to log EVERY iteration to a CSV file.
  This allows for plotting convergence curves later.
"""

from autogen import AssistantAgent, UserProxyAgent
from llms import llm_config
import sys
import os

# === Custom Logger ===
# Keep this to capture console output to file as backup
_original_stdout = sys.stdout
_original_stderr = sys.stderr

class Logger(object):
    """
    Logger that writes specific key info to benchmark_log.txt
    while keeping the console alive.
    """
    def __init__(self, filename="benchmark_log.txt"):
        self.terminal = _original_stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message: str):
        self.terminal.write(message)
        # Filter: only log important info to keep the text file clean
        important = (
            ("FINAL_RESULT:" in message)
            or ("Traceback" in message)
            or ("Error" in message)
            or ("ERROR" in message)
        )
        if important:
            self.log.write(message)
            self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger()
sys.stderr = sys.stdout

# === System prompts ===

# Updated Orchestrator: Now explicitly asks for CSV logging
coding_orchestrator = """
You are a Senior Hydrogeology Optimization Expert (Python).

OBJECTIVE:
Calibrate a MODFLOW groundwater model to find optimal values for 7 GEOLOGICAL PARAMETERS
to minimize MAE (mean absolute error).

PARAMETERS (Order is critical):
1. 'K_North'   in [0.1, 100.0]
2. 'K_South'   in [0.1, 100.0]
3. 'K_Chan_Up' in [0.1, 100.0]
4. 'K_Chan_Dn' in [0.1, 100.0]
5. 'K_Clay'    in [0.001, 1.0]
6. 'K_Sand'    in [1.0, 100.0]
7. 'K_Fault'   in [1.0, 100.0]

CODING TASK:
Write a COMPLETE Python script (single file) that performs the following:

1) Setup Logging (CRITICAL NEW STEP):
   - Before starting optimization, create a CSV file named 'calibration_history_case1.csv'.
   - Write the header row: ['Iteration', 'K_North', 'K_South', 'K_Chan_Up', 'K_Chan_Dn', 'K_Clay', 'K_Sand', 'K_Fault', 'MAE', 'RMSE']
   - Use a global counter to track iterations.

2) Simulation Interface:
   - Import: `from functions import simulation_objective_calibration`
   - Define a wrapper `objective_wrapper(x)`.
   - Inside the wrapper:
     a. Map `x` to the 7 parameters.
     b. Call `series, mse, mae = simulation_objective_calibration(params_dict)`.
     c. Calculate `rmse = np.sqrt(mse)`.
     d. Increment global iteration counter.
     e. APPEND the iteration data (iteration_num, params..., mae, rmse) to 'calibration_history_case1.csv'.
     f. Return `mae` (the optimizer minimizes this).

3) Optimizer Configuration:
   - Use `scipy.optimize.differential_evolution`.
   - Bounds: As defined above.
   - Settings: 
       maxiter = 50
       popsize = 10
       disp = True   (Allow DE to print progress to console)
       polish = True

4) Final Output:
   - Print exactly ONE final summary line starting with:
     FINAL_RESULT: K_North=..., ..., MAE=...

CONSTRAINTS:
- Ensure the CSV file is flushed/closed after every write so data isn't lost if the script crashes.
- Do NOT use 'validation_mode' or other unknown arguments. Keep the interface call simple.
"""

coding_executor = """
You are the Code Executor.
1. Execute the Python code provided.
2. If code fails, print the FULL Traceback.
3. Monitor output: if "FINAL_RESULT:" is found, reply "TERMINATE".
"""

# === Main run logic ===

def run_single_trial(trial_id: int) -> None:
    print(f"\n\n========== RUNNING CASE 1 TRIAL {trial_id} WITH LOGGING ==========")
    
    # Clean up old history if exists to avoid appending to old runs
    if os.path.exists("calibration_history_case1.csv"):
        try:
            os.remove("calibration_history_case1.csv")
            print(">>> Deleted old 'calibration_history_case1.csv'. Starting fresh.")
        except:
            pass

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
            "last_n_messages": 1,
            "work_dir": ".",
            "use_docker": False,
        },
    )

    user.initiate_chat(
        coder,
        message="Start calibration. IMPORTANT: Implement the CSV logging mechanism for every iteration!"
    )

if __name__ == "__main__":
    # Check dependencies
    if not os.path.exists("obs_data.csv"):
        print(">>> Warning: 'obs_data.csv' not found. Running setup_truth.py...")
        if os.path.exists("setup_truth.py"):
            os.system("python setup_truth.py")
        else:
            print("ERROR: setup_truth.py missing.")
            sys.exit(1)
            
    run_single_trial(1)
