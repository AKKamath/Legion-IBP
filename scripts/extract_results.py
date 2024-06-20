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
workloads = sys.argv[2].split()
types = sys.argv[3].split()

costs = {}
avg = {}
stddev = {}

for test in types:
    costs[test] = {}
    avg[test] = {}
    stddev[test] = {}

for test in types:
    for workload in workloads:
        training_file = os.path.join(folder_path + "/" + workload, 'training_' + test + '.log')
        if(os.path.exists(training_file)):
            costs[test][workload] = extract_costs(training_file)
            avg[test][workload] = np.mean(costs[test][workload])
            stddev[test][workload] = np.std(costs[test][workload])


print("Avg time (s)", end="\t")
for workload in workloads:
    print("{:s}".format(workload), end='\t')
print()
for i in types:
    print(i, end='\t')
    for workload in workloads:
        if i in costs:
            if workload in costs[i]:
                print("{:.4f}".format(avg[i][workload]), end='\t')
            else:
                print("NA", end="\t")
        else:
            print("NA", end="\t")
    print()
print()

print("Stddev", end='\t')
for workload in workloads:
    print("{:s}".format(workload), end='\t')
print()
for i in types:
    print(i, end='\t')
    for workload in workloads:
        if i in costs:
            if workload in costs[i]:
                print("{:.4f}".format(stddev[i][workload]), end='\t')
            else:
                print("NA", end="\t")
        else:
            print("NA", end="\t")
    print()