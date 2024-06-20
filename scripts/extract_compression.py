import re
import os
import sys
import numpy as np
import glob

RATIOS = {}
THPUTS = {}
def extract_ratio(file_path):
    dataset = re.findall(r'results/(.*)/sampling_comptest.log', file_path)[0]
    with open(file_path, 'r') as file:
        content = file.read()
        matches = re.findall(r'(.*): Uncompressed bytes: [0-9]+, compressed bytes: [0-9]+, ratio: ([0-9]+\.[0-9]+)', content)
        for match in matches:
            if match[0] not in RATIOS:
                RATIOS[match[0]] = {}
            RATIOS[match[0]][dataset] = float(match[1])

def extract_thput(file_path):
    dataset = re.findall(r'results/(.*)/sampling_comptest.log', file_path)[0]
    with open(file_path, 'r') as file:
        content = file.read()
        matches = re.findall(r'(.*): Time taken to decompress: ([0-9]+\.[0-9]+) ms. Throughput: ([0-9]+\.[0-9]+) MB/s', content)
        for match in matches:
            if match[0] not in THPUTS:
                THPUTS[match[0]] = {}
            THPUTS[match[0]][dataset] = float(match[2])

# Example usage
folder_path = sys.argv[1]
datasets = sys.argv[2].split()

for dataset in datasets:
    extract_ratio(folder_path + '/' + dataset + '/sampling_comptest.log')
    extract_thput(folder_path + '/' + dataset + '/sampling_comptest.log')


print("Ratios", end='\t')
#first_key = list(RATIOS.keys())[0]
#datasets = RATIOS[first_key]
for dataset in datasets:
    print("{:s}".format(dataset), end='\t')
print()

for algo in RATIOS:
    print("{:s}".format(algo), end='\t')
    for dataset in datasets:
        if(dataset in RATIOS[algo]):
            print("{:.2f}".format(RATIOS[algo][dataset]), end='\t')
        else:
            print("NA", end='\t')
    print()
print()

print("Thput", end='\t')
#first_key = list(THPUTS.keys())[0]
for dataset in datasets:
    print("{:s}".format(dataset), end='\t')
print()

for algo in THPUTS:
    print("{:s}".format(algo), end='\t')
    for dataset in datasets:
        if(dataset in THPUTS[algo]):
            print("{:.2f}".format(THPUTS[algo][dataset] / 1024), end='\t')
        else:
            print("NA", end='\t')
    print()
print()

print("Norm. thput", end='\t')
first_key = list(THPUTS.keys())[0]
for dataset in datasets:
    print("{:s}".format(dataset), end='\t')
print()

for algo in THPUTS:
    print("{:s}".format(algo), end='\t')
    for dataset in datasets:
        if(dataset in THPUTS[algo]):
            print("{:.2f}".format(THPUTS[algo][dataset] / THPUTS["Transfer"][dataset]), end='\t')
        else:
            print("NA", end='\t')
    print()