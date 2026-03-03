import os
import subprocess
import time

def run_exe():
    """
    Run TOUGHREACT executable with timeout protection.
    """
    exe_name = 'tr2_eos9.exe'
    
    # Check if executable exists
    if not os.path.exists(exe_name):
        print(f"Error: {exe_name} not found!")
        return False

    try:
        # Run the process with a 120-second timeout
        # input=b'\n' simulates pressing Enter
        # stdout/stderr=DEVNULL suppresses console output
        result = subprocess.run(
            exe_name,
            input=b'\n',
            timeout=120,  # Increased timeout to 120 seconds
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL,
            shell=True
        )
        
        # Check return code
        if result.returncode == 0:
            return True
        else:
            return False

    except subprocess.TimeoutExpired:
        # Kill the process if it hangs
        print("!!! TOUGHREACT Timed Out (Process Killed) !!!")
        os.system(f"taskkill /f /im {exe_name} >nul 2>&1")
        return False
        
    except Exception as e:
        print(f"Execution failed: {e}")
        return False

if __name__=="__main__":
    run_exe()
