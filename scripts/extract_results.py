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
training_comp_file = os.path.join(folder_path, 'training_comp.log')
training_mod_file = os.path.join(folder_path, 'training_mod.log')
training_file = os.path.join(folder_path, 'training.log')

comp_cost = extract_costs(training_comp_file)
mod_cost = extract_costs(training_mod_file)
def_cost = extract_costs(training_file)

# Calculate average cost
comp_costs = [cost for cost in comp_cost if cost is not None]
comp_avg = np.mean(comp_costs)
comp_stddev = np.std(comp_costs)

# Calculate average cost
mod_costs = [cost for cost in mod_cost if cost is not None]
mod_avg = np.mean(mod_costs)
mod_stddev = np.std(mod_costs)

def_costs = [cost for cost in def_cost if cost is not None]
def_avg = np.mean(def_costs)
def_stddev = np.std(def_costs)
print("{:s}\tAvg\tStddev".format(workload))
print("Baseline\t{:f}\t{:f}".format(def_avg, def_stddev))
print("Modified\t{:f}\t{:f}".format(mod_avg, mod_stddev))
print("Compressed\t{:f}\t{:f}".format(comp_avg, comp_stddev))