import re
import os
import sys
import numpy as np

def extract_breakdown(file_path):
    with open(file_path, 'r') as file:
        # Epoch 9 Batches: 24, train time : 286045.646 us, wait time : 531682.148 us
        content = file.read()
        matches = re.findall(r'train time : (\d+\.\d+) us', content)
        train_time = [float(match) for match in matches]
        wait_time = re.findall(r'wait time : (\d+\.\d+) us', content)
        wait_time = [float(match) for match in wait_time]
        return train_time[1:], wait_time[1:]  # Skip first element (warmup)

# Example usage
folder_path = sys.argv[1]
workloads = sys.argv[2].split()
types = sys.argv[3].split()
titles = None
if len(sys.argv) > 4:
    titles = sys.argv[4].split()

costs = {}
avg_train = {}
avg_wait = {}

for test in types:
    costs[test] = {}
    avg_train[test] = {}
    avg_wait[test] = {}

for test in types:
    for workload in workloads:
        training_file = os.path.join(folder_path + "/" + workload, 'training_' + test + '.log')
        if(os.path.exists(training_file)):
            train_time, wait_time = extract_breakdown(training_file)
            costs[test][workload] = (train_time, wait_time)
            # Convert to milliseconds from us
            avg_train[test][workload] = np.mean(train_time) / 1000
            avg_wait[test][workload] = np.mean(wait_time) / 1000

'''
print("Avg time (ms)", end="\t")
for workload in workloads:
    print("{:s}\t".format(workload), end='\t')
print()
print("", end="\t")
for workload in workloads:
    print("Train\tWait", end='\t')
print()
for i in types:
    print(i, end='\t')
    for workload in workloads:
        if i in costs:
            if workload in costs[i]:
                print("{:.4f}\t{:.4f}".format(avg_train[i][workload], avg_wait[i][workload]), end='\t')
            else:
                print("NA\tNA", end="\t")
        else:
            print("NA\tNA", end="\t")
    print()
print()
'''

print("Avg time (ms)", end="\t")
print("Train\tWait")
for workload in workloads:
    for i in types:
        print(workload + "(" + i + ")", end='\t')
        if i in costs:
            if workload in costs[i]:
                print("{:.4f}\t{:.4f}".format(avg_train[i][workload], avg_wait[i][workload]))
            else:
                print("NA\tNA")
        else:
            print("NA\tNA")
    print()
print()