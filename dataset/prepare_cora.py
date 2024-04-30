from ogb.nodeproppred import NodePropPredDataset
import numpy as np
from scipy.sparse import coo_matrix
from dgl import data, DGLGraph

dataset = data.CoraFullDataset()
graph = dataset[0] # graph: library-agnostic graph object
label = graph.ndata['label'].numpy()
feature = graph.ndata['feat'].numpy()
print(graph)

num_train = graph.num_nodes() * 0.4
num_val   = graph.num_nodes() * 0.1
num_test  = graph.num_nodes() * 0.5

train_idx, valid_idx, test_idx = np.split(np.arange(graph.num_nodes()), [int(num_train), int(num_train+num_val)])

print(len(train_idx), len(valid_idx), len(test_idx))

trainset = train_idx.astype(np.int32)
trainset.tofile('./cora/'+'trainingset')
validset = valid_idx.astype(np.int32)
validset.tofile('./cora/'+'validationset')
testset = test_idx.astype(np.int32)
testset.tofile('./cora/'+'testingset')
labels = label.astype(np.int32)
labels.tofile('./cora/'+'labels')
features = feature.astype(np.float32)
features.tofile('./cora/'+'features')


# Edge index in COO format
#edge_index = graph['edge_index']
#num_nodes = graph['num_nodes']
#print(num_nodes)
# Convert to COO matrix
#coo = coo_matrix((edge_index[1], (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes))

# Convert to CSR format
#csr = graph.to_scipy_sparse_matrix('csr') #coo.tocsr()

# Get the CSR row and col arrays
num_nodes = graph.num_nodes()
print(num_nodes)
# Convert to COO matrix
coo = coo_matrix((graph.all_edges()[1].numpy(), (graph.all_edges()[0].numpy(), graph.all_edges()[1].numpy())), shape=(num_nodes, num_nodes))

# Convert to CSR format
csr = coo.tocsr()

# Get the CSR row and col arrays
edge_src = (csr.indptr).astype(np.int64)
edge_src.tofile('./cora/'+'edge_src')
edge_dst = (csr.indices).astype(np.int32)
edge_dst.tofile('./cora/'+'edge_dst')

print(len(edge_src))
print(len(edge_dst))

#xtraformat = np.array(edge_index.T.flatten().tolist())
#xtraformat = xtraformat.astype(np.int32)
#xtraformat.tofile('./xtrapulp/cora_xtraformat')

