import os
import json

def count_ideas_in_directory(directory):
    total_ideas = 0
    print ("total number of files: ", len(os.listdir(directory)))
    
    for filename in os.listdir(directory):
        if filename.endswith('.json') and 'CSS' not in filename:
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                data = json.load(file)
                if 'ideas' in data:
                    total_ideas += len(data['ideas'])

    return total_ideas

# Example usage
directory_path = '../cache_results_claude_july/ideas_emnlp_dedup'
total_count = count_ideas_in_directory(directory_path)
print(f'Total number of ideas: {total_count}')

'''
../cache_results_claude_may/ideas_5k_dedup: N = 6496 
../cache_results_claude_july/ideas_emnlp_dedup: N = 7966
'''