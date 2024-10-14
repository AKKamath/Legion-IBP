import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
import dataset_info as di
import numpy as np
import torch
import dgl

def load(dataset_path, dataset_name):
    print(f"Opening dataset {dataset_name} at {dataset_path}")
    path, vertices_num, edges_num, features_dim, train_set_num, valid_set_num, \
        test_set_num, feat_dataset_file = di.get(dataset_path, dataset_name)

    # Load graph
    edge_src_path = path + "edge_src"
    edge_dst_path = path + "edge_dst"

    csr_node_index = np.fromfile(
        edge_src_path,
        dtype="int64",
    )
    csr_node_index = torch.from_numpy(csr_node_index).share_memory_().pin_memory()
    print(csr_node_index.shape)

    csr_dst_ids = np.fromfile(
        edge_dst_path,
        dtype="int32",
    )
    csr_dst_ids = torch.from_numpy(csr_dst_ids).share_memory_().pin_memory()
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
    training_ids = training_ids.type(torch.int64).share_memory_().pin_memory()
    validation_ids = torch.from_numpy(validation_ids)
    validation_ids = validation_ids.type(torch.int64).share_memory_().pin_memory()
    testing_ids = torch.from_numpy(testing_ids)
    testing_ids = testing_ids.type(torch.int64).share_memory_().pin_memory()
    
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
        labels = torch.from_numpy(labels).share_memory_().pin_memory()
        features_path = feat_dataset_file + "features"
        features_copy = np.fromfile(
            features_path,
            dtype="float32",
        )
        # Copy features into main graph
        features = np.tile(features_copy, (vertices_num * features_dim // features_copy.shape[0]))
        features = np.append(features, features_copy[:(vertices_num * features_dim) % features_copy.shape[0]])
    else:
        labels = np.fromfile(
            labels_path,
            dtype="int32",
        )
        labels = torch.from_numpy(labels).share_memory_().pin_memory()

        features = np.fromfile(
            features_path,
            dtype="float32",
        )
    print("Loaded feats")
    features = torch.from_numpy(features).share_memory_()
    print("Shared feats")
    #features = features.pin_memory()
    #print("Pinned feats")
    features = features.reshape((vertices_num, features_dim))
    print("Modded feats")
    g = dgl.graph(('csr', (csr_node_index, csr_dst_ids, [])))
    g.ndata["feat"] = features
    g.ndata["label"] = labels
    print("Setup graph")
    return g, features, labels, training_ids, validation_ids, testing_ids
    exit()