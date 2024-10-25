import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
import dataset_info as di
import numpy as np
import torch
import dgl

def pin_inplace(tensor):
    try:
        cudart = torch.cuda.cudart()
        r = cudart.cudaHostRegister(tensor.data_ptr(), tensor.numel() * tensor.element_size(), 0)
        assert tensor.is_pinned()
    except Exception as e:
        print(f"Failed to pin tensor: {e}")
    return tensor

def load(dataset_path, dataset_name):
    print(f"Opening dataset {dataset_name} at {dataset_path}")
    path, vertices_num, edges_num, features_dim, train_set_num, valid_set_num, \
        test_set_num, feat_dataset_file = di.get(dataset_path, dataset_name)

    # Load graph
    edge_src_path = path + "edge_src"
    edge_dst_path = path + "edge_dst"

    csr_node_index_file = np.fromfile(
        edge_src_path,
        dtype="int64",
    )
    # If vertices don't divide evenly, pad with last value
    csr_node_index = torch.empty((vertices_num + 1), dtype=torch.int64).share_memory_()
    csr_node_index[:csr_node_index_file.shape[0]] = torch.from_numpy(csr_node_index_file)
    csr_node_index[csr_node_index_file.shape[0]:] = csr_node_index_file[-1]
    print(csr_node_index.shape)

    csr_dst_ids = np.fromfile(
        edge_dst_path,
        dtype="int32",
    )
    csr_dst_ids = torch.from_numpy(csr_dst_ids).share_memory_()
    csr_dst_ids = csr_dst_ids.type(torch.int64)
    
    training_path = path  + "trainingset"
    validation_path = path  + "validationset"
    testing_path = path  + "testingset"
    features_path = path + "features"
    labels_path = path + "labels"

    training_ids = np.fromfile(
        training_path,
        dtype="int32",
    )
    validation_ids = np.fromfile(
        validation_path,
        dtype="int32",
    )
    testing_ids = np.fromfile(
        testing_path,
        dtype="int32",
    )

    training_ids = torch.from_numpy(training_ids)
    training_ids = training_ids.type(torch.int64).share_memory_()
    validation_ids = torch.from_numpy(validation_ids)
    validation_ids = validation_ids.type(torch.int64).share_memory_()
    testing_ids = torch.from_numpy(testing_ids)
    testing_ids = testing_ids.type(torch.int64).share_memory_()
    
    print("Loaded all here")
    if feat_dataset_file != "":
        labels_path = feat_dataset_file + "labels"
        # Get new labels
        labels_copy = np.fromfile(
            labels_path,
            dtype="int32",
        )
        # Copy labels into main graph
        labels = np.tile(labels_copy, (vertices_num // labels_copy.shape[0]))
        labels = np.append(labels, labels_copy[:(vertices_num) % labels_copy.shape[0]])
        labels = torch.from_numpy(labels).share_memory_()
        features_path = feat_dataset_file + "features"
        features_copy = np.fromfile(
            features_path,
            dtype="float32",
        )
        features = torch.empty((vertices_num * features_dim), dtype=torch.float32).share_memory_()
        for i in range(vertices_num * features_dim // features_copy.shape[0]):
            features[i * features_copy.shape[0]:(i + 1) * features_copy.shape[0]] = torch.from_numpy(features_copy)
        features[(vertices_num * features_dim) - (vertices_num * features_dim) % features_copy.shape[0]:] = \
            torch.from_numpy(features_copy[:(vertices_num * features_dim) % features_copy.shape[0]])
        # Copy features into main graph
        #features = np.tile(features_copy, (vertices_num * features_dim // features_copy.shape[0]))
        #features = np.append(features, features_copy[:(vertices_num * features_dim) % features_copy.shape[0]])
    else:
        labels = np.fromfile(
            labels_path,
            dtype="int32",
        )
        labels = torch.from_numpy(labels).share_memory_()

        features = np.fromfile(
            features_path,
            dtype="float32",
        )
        print("Loaded feats")
        features = torch.from_numpy(features).share_memory_()
    # Reshape and pin features
    features = pin_inplace(features.reshape((vertices_num, features_dim)))
    print(f"Features shared? {features.is_shared()}; pinned? {features.is_pinned()}")
    g = dgl.graph(('csr', (csr_node_index, csr_dst_ids, [])))
    # Reverse graph as DGL uses "in" edges for sampling, while Legion uses "out"
    g = dgl.reverse(g, copy_edata=False, copy_ndata=False)
    g.ndata["feat"] = features
    g.ndata["label"] = labels
    print("Setup graph")
    return g, features, labels, training_ids, validation_ids, testing_ids
    exit()