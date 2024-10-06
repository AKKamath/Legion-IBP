import os 
import argparse 
import subprocess
import re
import networkx as nx
import math

def parse_topo_output(output):
    """
    Parses the output from `nvidia-smi topo -m` to extract the NVLink connections between GPUs.
    This function is adjusted based on your provided example output.
    """
    connections = []
    lines = output.splitlines()
    gpu_lines = [line for line in lines if line.startswith("GPU")]
    for i, line in enumerate(gpu_lines):
        elements = line.split()
        for j, elem in enumerate(elements[1:], start=0):  # Start from the first GPU column
            if elem.startswith("NV"):
                connections.append((i, j))
    return connections

def get_nvlink_topology():
    # Execute the `nvidia-smi topo -m` command to get the topology matrix
    result = subprocess.run(['nvidia-smi', 'topo', '-m'], stdout=subprocess.PIPE, text=True)
    connections = parse_topo_output(result.stdout)
    return connections

def find_largest_fully_connected_group(G):
    """
    Finds the largest fully connected group (clique) in the graph and returns its size
    and the list of such groups if there are multiple of the same size.
    """
    cliques = list(nx.find_cliques(G))
    max_size = max(len(clique) for clique in cliques) if cliques else 1
    max_cliques = [clique for clique in cliques if len(clique) == max_size]
    return max_size, max_cliques
        
def Run(args):
    import dataset_info as di
    path, vertices_num, edges_num, features_dim, train_set_num, valid_set_num, \
        test_set_num, feat_dataset_file = di.get(args.dataset_path, args.dataset_name)
    with open("meta_config","w") as file:
        file.write("{} {} {} {} {} {} {} {} {} {} {}".format(path, args.train_batch_size, vertices_num, edges_num, features_dim, train_set_num, valid_set_num, test_set_num, args.cache_memory, args.epoch, feat_dataset_file))

    gpu_number = args.gpu_number
    
    if args.usenvlink == 1:
        connections = get_nvlink_topology()
        G = nx.Graph()
        G.add_edges_from(connections)
        group_size, fully_connected_groups = find_largest_fully_connected_group(G)
        if fully_connected_groups or group_size == 1:
            print(f"NVLink clique size: {group_size}, Number of NVLink cliques: {int(gpu_number/group_size)}")
        cache_agg_mode = math.log2(group_size)
    else:
        cache_agg_mode = 0

    # get the current file path
    current_file_path = os.path.abspath(__file__)

    # get the Legion_home path
    Legion_home = os.path.dirname(current_file_path)

    server_path = os.path.join(Legion_home, "sampling_server/build/bin/sampling_server {} {} {}").format(gpu_number, cache_agg_mode, args.dyn_cache)
    os.system(server_path)
    ## TODO, integrate Legion server in python module


if __name__ == "__main__":

    argparser = argparse.ArgumentParser("Legion Server.")
    argparser.add_argument('--dataset_path', type=str, default="./dataset")
    argparser.add_argument('--dataset_name', type=str, default="ukunion")
    argparser.add_argument('--train_batch_size', type=int, default=8000)
    argparser.add_argument('--fanout', type=list, default=[25, 10])
    argparser.add_argument('--gpu_number', type=int, default=2)
    argparser.add_argument('--epoch', type=int, default=2)
    argparser.add_argument('--cache_memory', type=int, default=38000000)
    argparser.add_argument('--usenvlink', type=int, default=1)
    argparser.add_argument('--dyn_cache', type=int, default=0)
    args = argparser.parse_args()

    Run(args)
