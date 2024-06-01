import subprocess
import os
import time
def run_script(script_path):
    startTime = time.time()
    try:
        # Run the script and capture the output
        result = subprocess.run(
            ['python', script_path], 
            capture_output=True, 
            text=True,
            check=True
        )
        # Return the output
        endTime = time.time()
        executionTime = endTime - startTime
        return result.stdout, result.stderr, executionTime
    except subprocess.CalledProcessError as e:
        # Return the error output if the script fails
        return e.stdout, e.stderr
    
