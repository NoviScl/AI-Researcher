import subprocess
import os
from tqdm import tqdm 

# The path to the python program you want to run
cache_dir = '../cache_results_claude_may/execution/factuality_prompting_method_prompting/'
filenames = os.listdir(os.path.join(cache_dir))

if "utils.py" not in filenames:
    ## copy over the utils file 
    os.system("cp prompts/utils.py " + cache_dir)

total_files = 0
success_count = 0
for python_program_path in tqdm(filenames):
    if python_program_path == "utils.py" or (not python_program_path.endswith(".py")):
        continue
    print ("working on idea: ", python_program_path)
    total_files += 1
    python_program_path = os.path.join(cache_dir, python_program_path)

    with open(python_program_path.replace(".py", "_log.txt"), 'w') as log_file:
        # Use subprocess.run to execute the python program
        result = subprocess.run(['python3', python_program_path], stdout=log_file, stderr=subprocess.STDOUT)

    # # Print the standard output and standard error of the program
    # print ("STDOUT:", result.stdout)
    # print ("STDERR:", result.stderr)

    # Check if the program executed successfully
    if result.returncode == 0:
        print("The program executed successfully.")
        success_count += 1
    else:
        print("The program encountered an error.", "Error code:", result.returncode)
    
    print("\n")

print ("execution success rate: {} / {} = {}%".format(success_count, total_files, success_count / total_files * 100))