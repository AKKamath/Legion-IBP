from ogb.nodeproppred import NodePropPredDataset
import numpy as np
from scipy.sparse import coo_matrix
from dgl import data, DGLGraph
import dgl
import torch

# root = '/home/wzq/datasets/OGB' can be replaced to the path where you store the dataset
dataset = data.RedditDataset()
g = dataset[0]
print(g)

train_idx = torch.nonzero(g.ndata['train_mask'], as_tuple=False).squeeze().numpy()
valid_idx = torch.nonzero(g.ndata['val_mask'], as_tuple=False).squeeze().numpy()
test_idx = torch.nonzero(g.ndata['test_mask'], as_tuple=False).squeeze().numpy()

print(len(train_idx))
print(len(valid_idx))
print(len(test_idx))

label = g.ndata['label'].numpy()
feats = g.ndata['feat'].numpy()

trainset = train_idx.astype(np.int32)
trainset.tofile('./reddit/'+'trainingset')
validset = valid_idx.astype(np.int32)
validset.tofile('./reddit/'+'validationset')
testset = test_idx.astype(np.int32)
testset.tofile('./reddit/'+'testingset')
labels = label.astype(np.int32)
labels.tofile('./reddit/'+'labels')
features = feats.astype(np.float32)
features.tofile('./reddit/'+'features')

# Edge index in COO format
#edge_index = graph['edge_index']
num_nodes = g.num_nodes()
print(num_nodes)
# Convert to COO matrix
coo = coo_matrix((g.all_edges()[1].numpy(), (g.all_edges()[0].numpy(), g.all_edges()[1].numpy())), shape=(num_nodes, num_nodes))

# Convert to CSR format
csr = coo.tocsr()

# Get the CSR row and col arrays
edge_src = (csr.indptr).astype(np.int64)
edge_src.tofile('./reddit/'+'edge_src')
edge_dst = (csr.indices).astype(np.int32)
edge_dst.tofile('./reddit/'+'edge_dst')

print(len(edge_src))
print(len(edge_dst))

# Edge index in COO format
#edge_index = graph['edge_index']
#num_nodes = graph['num_nodes']
#print(num_nodes)
# Convert to COO matrix
#coo = coo_matrix((edge_index[1], (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes))
'''
# Convert to CSR format
csr = coo.tocsr()

# Get the CSR row and col arrays
edge_src = (csr.indptr).astype(np.int64)
edge_src.tofile('./reddit/'+'edge_src')
edge_dst = (csr.indices).astype(np.int32)
edge_dst.tofile('./reddit/'+'edge_dst')

xtraformat = np.array(edge_index.T.flatten().tolist())
xtraformat = xtraformat.astype(np.int32)
xtraformat.tofile('./xtrapulp/reddit_xtraformat')
'''
