from ogb.lsc import MAG240MDataset
import numpy as np
from scipy.sparse import coo_matrix
import torch

# root = '/home/wzq/datasets/OGB' can be replaced to the path where you store the dataset
dataset = MAG240MDataset()
paper_offset = dataset.num_authors + dataset.num_institutions
num_nodes = paper_offset + dataset.num_papers
num_features = dataset.num_paper_features

split_idx = dataset.get_idx_split()
train_idx = (torch.LongTensor(split_idx["train"]) + paper_offset).numpy()
valid_idx = (torch.LongTensor(split_idx["valid"]) + paper_offset).numpy()
test_idx  = (torch.LongTensor(split_idx["test-dev"]) + paper_offset).numpy()
#graph, label = dataset[0] # graph: library-agnostic graph object
label = dataset.paper_label

print(dataset.paper_feat)

print(len(train_idx), len(valid_idx), len(test_idx))
print(len(label))

# Edge index in COO format
edge_index = dataset.edge_index('paper', 'cites', 'paper')
print(num_nodes)
print(len(edge_index))
print(len(edge_index[0]))
print(len(edge_index[1]))
'''
xtraformat = np.array(edge_index.T.flatten().tolist())
xtraformat = xtraformat.astype(np.int32)
xtraformat.tofile('./xtrapulp/mag_xtraformat')
'''
