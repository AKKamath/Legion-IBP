import re
import os
import sys
import numpy as np

#batch id: 999; 1000 batches, time per sample: 21730 us
def extract_samprate(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        matches = re.findall(r'batch id: \d+; (\d+) batches, time per sample: (\d+) us', content)
        batches=sum(int(match[0]) for match in matches)
        tot_time=sum(int(match[1]) for match in matches)
        return batches * 1e6 / tot_time

# Example usage
folder_path = sys.argv[1]
workloads = sys.argv[2].split()
types = sys.argv[3].split()

samprate = {}
avg = {}
stddev = {}

for test in types:
    samprate[test] = {}
    avg[test] = {}
    stddev[test] = {}

for test in types:
    for workload in workloads:
        training_file = os.path.join(folder_path + "/" + workload, 'sampling_' + test + '.log')
        if(os.path.exists(training_file)):
            samprate[test][workload] = extract_samprate(training_file)


print("Rate (Batches / s)", end="\t")
for workload in workloads:
    print("{:s}".format(workload), end='\t')
print()
for i in types:
    print(i, end='\t')
    for workload in workloads:
        if i in samprate:
            if workload in samprate[i]:
                print("{:.2f}".format(samprate[i][workload]), end='\t')
            else:
                print("NA", end="\t")
        else:
            print("NA", end="\t")
    print()
print()