from ogb.lsc import MAG240MDataset
import numpy as np
from scipy.sparse import coo_matrix
import torch

# root = '/home/wzq/datasets/OGB' can be replaced to the path where you store the dataset
dataset = MAG240MDataset()
paper_offset = dataset.num_authors + dataset.num_institutions
num_nodes = paper_offset + dataset.num_papers
num_features = dataset.num_paper_features
full_feat = np.memmap(
        './mag/full_feat',
        mode="w+",
        dtype="float16",
        shape=(
            dataset.num_authors + dataset.num_institutions + dataset.num_papers,
            dataset.num_paper_features,
        ),
    )

split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = torch.LongTensor(split_idx["train"]) + paper_offset, torch.LongTensor(split_idx["valid"]) + paper_offset, torch.LongTensor(split_idx["test"]) + paper_offset
graph, label = dataset[0] # graph: library-agnostic graph object

trainset = train_idx.astype(np.int32)
trainset.tofile('./mag/'+'trainingset')
validset = valid_idx.astype(np.int32)
validset.tofile('./mag/'+'validationset')
testset = test_idx.astype(np.int32)
testset.tofile('./mag/'+'testingset')
labels = label.astype(np.int32)
labels.tofile('./mag/'+'labels')
features = graph['node_feat'].astype(np.float16)
features.tofile('./mag/'+'features')

# Edge index in COO format
edge_index = graph['edge_index']
num_nodes = graph['num_nodes']
print(num_nodes)
# Convert to COO matrix
coo = coo_matrix((edge_index[1], (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes))

# Convert to CSR format
csr = coo.tocsr()

# Get the CSR row and col arrays
edge_src = (csr.indptr).astype(np.int64)
edge_src.tofile('./mag/'+'edge_src')
edge_dst = (csr.indices).astype(np.int32)
edge_dst.tofile('./mag/'+'edge_dst')

xtraformat = np.array(edge_index.T.flatten().tolist())
xtraformat = xtraformat.astype(np.int32)
xtraformat.tofile('./xtrapulp/mag_xtraformat')

