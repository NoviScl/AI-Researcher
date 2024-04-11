import subprocess
import os
from tqdm import tqdm 

# The path to the python program you want to run
cache_dir = '../cache_results_claude/execution/factuality_prompting_new_method_prompting/'
filenames = os.listdir(os.path.join(cache_dir))

for python_program_path in tqdm(filenames):
    if python_program_path == "utils.py":
        continue
    print ("working on idea: ", python_program_path)
    python_program_path = os.path.join(cache_dir, python_program_path)

    # Use subprocess.run to execute the python program
    result = subprocess.run(['python3', python_program_path], capture_output=True, text=True)

    # Print the standard output and standard error of the program
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    # Check if the program executed successfully
    if result.returncode == 0:
        print("The program executed successfully.")
    else:
        print("The program encountered an error.", "Error code:", result.returncode)
    
    print("\n")