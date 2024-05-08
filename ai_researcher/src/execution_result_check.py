import os 

cache_dir = '../cache_results_claude_may/execution/safety_prompting_method_prompting/'
filenames = os.listdir(os.path.join(cache_dir))

def parse_log_file(log_file):
    baseline_accuracy = -1
    proposed_accuracy = -1
    style_accuracy = -1

    lines = [line.strip() for line in log_file if len(line.strip()) > 0]
    lines = lines[-3 : ]
    for line in lines:
        if ':' in line:
            key = line.split(':')[0]
            value = line.split(':')[-1]
            if 'baseline' in key.lower():
                baseline_accuracy = float(value.strip())
            elif 'proposed' in key.lower():
                proposed_accuracy = float(value.strip())
            elif 'style' in key.lower():
                style_accuracy = float(value.strip())
            
    return baseline_accuracy, proposed_accuracy, style_accuracy

counter = 0
style_pass = 0
proposed_better = 0
for filename in filenames:
    if filename.endswith("_log.txt"):
        with open(os.path.join(cache_dir, filename), 'r') as log_file:
            log = log_file.readlines()
        
        baseline_accuracy, proposed_accuracy, style_accuracy = parse_log_file(log)
        if baseline_accuracy >= 0 and proposed_accuracy >= 0 and style_accuracy >= 0:
            counter += 1
            if style_accuracy == 1.:
                style_pass += 1
                if proposed_accuracy > baseline_accuracy:
                    proposed_better += 1

print (counter)
print (style_pass)
print (proposed_better)
