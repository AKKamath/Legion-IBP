from ogb.lsc import MAG240MDataset
import numpy as np
from scipy.sparse import coo_matrix
import torch

# root = '/home/wzq/datasets/OGB' can be replaced to the path where you store the dataset
dataset = MAG240MDataset()
#paper_offset = dataset.num_authors + dataset.num_institutions
num_nodes = dataset.num_papers
num_features = dataset.num_paper_features

print(dataset)

split_idx = dataset.get_idx_split()
train_idx = (torch.LongTensor(split_idx["train"])).numpy()
valid_idx = (torch.LongTensor(split_idx["valid"])).numpy()
test_idx  = (torch.LongTensor(split_idx["test-dev"])).numpy()
#graph, label = dataset[0] # graph: library-agnostic graph object
label = dataset.paper_label

trainset = train_idx.astype(np.int32)
trainset.tofile('./mag/'+'trainingset')
validset = valid_idx.astype(np.int32)
validset.tofile('./mag/'+'validationset')
testset = test_idx.astype(np.int32)
testset.tofile('./mag/'+'testingset')
labels = label.astype(np.int32)
labels.tofile('./mag/'+'labels')
#features = .astype(np.float16)
dataset.paper_feat.tofile('./mag/'+'features')


# Edge index in COO format
edge_index = dataset.edge_index('paper', 'cites', 'paper')
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
'''
xtraformat = np.array(edge_index.T.flatten().tolist())
xtraformat = xtraformat.astype(np.int32)
xtraformat.tofile('./xtrapulp/mag_xtraformat')
'''
