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

    feat_dataset_file = ""
    if args.dataset_name == "products":
        path =  args.dataset_path + "/products/"
        vertices_num = 2449029
        edges_num = 123718280
        features_dim = 100
        train_set_num = 196615
        valid_set_num = 39323
        test_set_num = 2213091
    elif args.dataset_name == "paper100m":
        path = args.dataset_path + "/paper100m/"
        vertices_num = 111059956
        edges_num = 1615685872
        features_dim = 128
        train_set_num = 1207179  
        valid_set_num = 125265
        test_set_num = 214338
    elif args.dataset_name == "com-friendster":
        path = args.dataset_path + "/com-friendster/"
        vertices_num = 65608366
        edges_num = 1806067135
        features_dim = 256
        train_set_num = 6560836
        valid_set_num = 100000
        test_set_num = 100000
    elif args.dataset_name == "ukunion":
        path = args.dataset_path + "/ukunion/"
        vertices_num = 133633040
        edges_num = 5507679822
        features_dim = 116
        train_set_num = 13363304
        valid_set_num = 100000
        test_set_num = 100000
    elif args.dataset_name == "uk2014":
        path = args.dataset_path + "/uk2014/"
        vertices_num = 787801471
        edges_num = 47284178505
        features_dim = 128
        train_set_num = 78780147
        valid_set_num = 100000
        test_set_num = 100000
    elif args.dataset_name == "clueweb":
        path = args.dataset_path + "/clueweb/"
        vertices_num = 955207488
        edges_num = 42574107469
        features_dim = 128
        train_set_num = 95520748
        valid_set_num = 100000
        test_set_num = 100000
    elif args.dataset_name == "cora":
        path = args.dataset_path + "/cora/"
        vertices_num = 19793
        edges_num = 126842
        features_dim = 8710
        train_set_num = 7917
        valid_set_num = 1979
        test_set_num = 9897
    elif args.dataset_name == "reddit":
        path = args.dataset_path + "/reddit/"
        vertices_num = 232965
        edges_num = 114615892
        features_dim = 602
        train_set_num = 153431
        valid_set_num = 23831
        test_set_num = 55703
    elif args.dataset_name == "pubmed":
        path = args.dataset_path + "/pubmed/"
        vertices_num = 19717
        edges_num = 88651
        features_dim = 500
        train_set_num = 7886
        valid_set_num = 1972
        test_set_num = 9859
    elif args.dataset_name == "mag":
        path = args.dataset_path + "/mag/"
        vertices_num = 121751666
        edges_num = 1297748926
        features_dim = 384
        train_set_num = 1112392
        valid_set_num = 138949
        test_set_num = 88092
    elif args.dataset_name == "citeseer":
        path = args.dataset_path + "/citeseer/"
        vertices_num = 3327
        edges_num = 9228
        features_dim = 3703
        train_set_num = 1330
        valid_set_num = 333
        test_set_num = 1664
    elif args.dataset_name == "pubmed_ls":
        path = args.dataset_path + "/ukunion/"
        vertices_num = 133633040
        edges_num = 5507679822
        features_dim = 500
        train_set_num = 13363304
        valid_set_num = 100000
        test_set_num = 100000
        feat_dataset_file = args.dataset_path + "/pubmed/"
    elif args.dataset_name == "citeseer_ls":
        path =  args.dataset_path + "/products/"
        vertices_num = 2449029
        edges_num = 123718280
        features_dim = 3703
        train_set_num = 196615
        valid_set_num = 39323
        test_set_num = 2213091
        feat_dataset_file = args.dataset_path + "/citeseer/"
    elif args.dataset_name == "cora_ls":
        path =  args.dataset_path + "/products/"
        vertices_num = 2449029
        edges_num = 123718280
        features_dim = 8710
        train_set_num = 196615
        valid_set_num = 39323
        test_set_num = 2213091
        feat_dataset_file = args.dataset_path + "/cora/"
    else:
        print("invalid dataset path")
        exit
    

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
