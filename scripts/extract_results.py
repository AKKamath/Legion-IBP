import re
import os
import sys
import numpy as np

def extract_costs(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        matches = re.findall(r'Cost:(\d+\.\d+)', content)
        costs = [float(match) for match in matches]
        return costs[1:]

# Example usage
folder_path = sys.argv[1]
workload = sys.argv[1].split('/')[-1]
types = sys.argv[2].split()

costs = {}
avg = {}
stddev = {}

for i in types:
    training_file = os.path.join(folder_path, 'training_' + i + '.log')
    if(os.path.exists(training_file)):
        costs[i] = extract_costs(training_file)
        avg[i] = np.mean(costs[i])
        stddev[i] = np.std(costs[i])

print("{:s}\tAvg\tStddev".format(workload))
for i in types:
    if i in costs:
        print("{:s}\t{:.4f}\t{:.4f}".format(i, avg[i], stddev[i]))
    else:
        print("{:s}\t\t".format(i))