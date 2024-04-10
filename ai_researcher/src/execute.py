import subprocess

# The path to the python program you want to run
python_program_path = '../cache_results_claude/execution/factuality_prompting_new_method_prompting/collaborative_reasoning_prompting.py'

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
