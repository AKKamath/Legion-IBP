import re
import os
import sys
import numpy as np

#batch id: 999; 1000 batches, time per sample: 21730 us
def extract_cache_cap(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        comp_match = re.findall(r'Inserted (\d+) compressed feats \(expected at least \d+\)', content)
        matches = re.findall(r'Initializing dynamic cache with (\d+) \(out of', content)

        orig = 0
        comp = 0

        if comp_match:
            comp = int(comp_match[0])

        if matches:
            orig = int(matches[0])
        return orig, comp

# Example usage
folder_path = sys.argv[1]
workloads = sys.argv[2].split()
types = sys.argv[3].split()

cache_cap = {}
avg = {}

for test in types:
    cache_cap[test] = {}
    avg[test] = {}

for test in types:
    for workload in workloads:
        sample_file = os.path.join(folder_path + "/" + workload, 'sampling_' + test + '.log')
        if(os.path.exists(sample_file)):
            cache_cap[test][workload] = extract_cache_cap(sample_file)


print("Cache Capacity", end="\t")
for workload in workloads:
    print("{:s}".format(workload), end="\t")
print()
print("Original", end="\t")
for workload in workloads:
    for i in types:
        if i in cache_cap:
            if workload in cache_cap[i]:
                print("{:d}".format(cache_cap[i][workload][0]), end='\t')
            else:
                print("NA", end="\t")
        else:
            print("NA", end="\t")
print()
print("Compressed", end="\t")
for workload in workloads:
    for i in types:
        if i in cache_cap:
            if workload in cache_cap[i]:
                print("{:d}".format(cache_cap[i][workload][1]), end='\t')
            else:
                print("NA", end="\t")
        else:
            print("NA", end="\t")
print()
print("Increase", end="\t")
for workload in workloads:
    for i in types:
        if i in cache_cap:
            if workload in cache_cap[i]:
                ratio = cache_cap[i][workload][1] / cache_cap[i][workload][0] if cache_cap[i][workload][0] > 0 else 0
                print("{:.2f}X".format(ratio), end='\t')
            else:
                print("NA", end="\t")
        else:
            print("NA", end="\t")
print()