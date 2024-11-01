import os

# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import sys
import tempfile
import argparse
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.amp import autocast
from torch.cuda.amp import GradScaler

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as Func

import dgl
from dgl.nn.pytorch import SAGEConv
from dgl.heterograph import DGLBlock
import time
import numpy as np
import torchmetrics
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler,
)
torch.set_printoptions(threshold=np.inf)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    if torch.cuda.is_available():
      dist.init_process_group('nccl', rank=rank, world_size=world_size)
    else:
      dist.init_process_group('gloo', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class SAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_feats, n_hidden, 'mean'))
        for _ in range(1, n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(SAGEConv(n_hidden, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

def create_dgl_block(src, dst, num_src_nodes, num_dst_nodes, fp16):
    gidx = dgl.heterograph_index.create_unitgraph_from_coo(2, num_src_nodes, num_dst_nodes, src, dst, 'coo', row_sorted=True)
    g = DGLBlock(gidx, (['_N'], ['_N']), ['_E'])

    # Convert format for fp16 training
    if fp16:
        g = g.formats(['coo', 'csc'])
        g.create_formats_()
    return g
scaler = GradScaler()

def backward(scaler, loss, optimizer):
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

total_train_time = 0
train_batches = 0
def train_one_step(model, optimizer, loss_fcn, device, feat_list, blocks, fp16, device_id, it):

    features = blocks[0].srcdata["feat"]
    labels = blocks[-1].dstdata["label"]

    data_type = torch.float32
    # Typecast fp32 to fp16
    if fp16:
        features = features.view(torch.float16)
        data_type = torch.float16
    #print(blocks)

    start = time.perf_counter_ns()
    with autocast('cuda', enabled=fp16, dtype=data_type):
        #print("Timestamp {:d}, sample {:d}, worker {:d}, system {:s}, {:s}".format(time.time_ns(), it, device_id, "trainer", "getsample"))
        batch_pred = model(blocks, features)
        long_labels = torch.as_tensor(labels, dtype=torch.long, device=device)
        loss = loss_fcn(batch_pred, long_labels)
    if fp16:
        backward(scaler, loss, optimizer)
    else:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    torch.cuda.synchronize()
    end = time.perf_counter_ns()    
    #print("Timestamp {:d}, sample {:d}, worker {:d}, system {:s}, {:s}".format(time.time_ns(), it, device_id, "trainer", "finishtrain"))
    global total_train_time
    global train_batches
    total_train_time += end - start
    train_batches += 1
    return 0

def valid_one_step(model, metric, device, feat_len, blocks, fp16):
    
    features = blocks[0].srcdata["feat"]
    labels = blocks[-1].dstdata["label"]

    data_type = torch.float32
    # Typecast fp32 to fp16
    if fp16:
        features = features.view(torch.float16)
        data_type = torch.float16
    
    start = time.perf_counter_ns()
    with autocast('cuda', enabled=fp16, dtype=data_type):
        batch_pred = model(blocks, features)
    long_labels = torch.as_tensor(labels, dtype=torch.long, device=device)
    batch_pred = torch.softmax(batch_pred, dim=1).to(device)
    acc = metric(batch_pred, long_labels)
    end = time.perf_counter_ns()
    global total_train_time
    global train_batches
    total_train_time += end - start
    train_batches += 1
    return acc

def test_one_step(model, metric, device, feat_len, blocks, fp16):
    
    features = blocks[0].srcdata["feat"]
    labels = blocks[-1].dstdata["label"]

    data_type = torch.float32
    # Typecast fp32 to fp16
    if fp16:
        features = features.view(torch.float16)
        data_type = torch.float16
    
    with autocast('cuda', enabled=fp16, dtype=data_type):
        batch_pred = model(blocks, features)
        long_labels = torch.as_tensor(labels, dtype=torch.long, device=device)
        batch_pred = torch.softmax(batch_pred, dim=1).to(device)
    acc = metric(batch_pred, long_labels)
    return acc

def worker_process(rank, world_size, args, g_data):
    print(f"Running GNN Training on CUDA {rank}.")
    g, train_idx_split, val_idx_split, test_idx_split, \
        train_step, valid_step, test_step = g_data

    train_idx = train_idx_split[rank]
    val_idx = val_idx_split[rank]
    test_idx = test_idx_split[rank]
    #train_idx = train_idx.split((train_idx.shape[0] + world_size - 1) // world_size)[rank]
    #val_idx = val_idx.split((val_idx.shape[0] + world_size - 1) // world_size)[rank]
    #test_idx = test_idx.split((test_idx.shape[0] + world_size - 1) // world_size)[rank]
    valid_batch_size = (((val_idx.shape[0] - 1)//valid_step + 1))
    test_batch_size = ((test_idx.shape[0] - 1)//test_step + 1)

    device_id = rank
    setup(rank, world_size)
    cuda_device = torch.device("cuda:{}".format(device_id))
    torch.cuda.set_device(cuda_device)
    #train_steps, valid_steps, test_steps = (train_idx.shape[0] + args.train_batch_size) // args.train_batch_size, \
    #    (val_idx.shape[0] + 511) // 512, (test_idx.shape[0] + 511) // 512

    #print(train_steps, valid_steps, test_steps)

    sampler = NeighborSampler(
        args.nbrs_num,  # fanout for [layer-0, layer-1, layer-2]
        prefetch_node_feats=["feat"],
        prefetch_labels=["label"],
    )
    train_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        device=cuda_device,
        batch_size=args.train_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        use_uva=True,
    )

    val_dataloader = DataLoader(
        g,
        val_idx,
        sampler,
        device=cuda_device,
        batch_size=valid_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        use_uva=True,
    )

    test_dataloader = DataLoader(
        g,
        test_idx,
        sampler,
        device=cuda_device,
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        use_uva=True,
    )

    feat_len = args.features_num

    # Multiply by 2 if float16, because each fp32 gets converted to 2 fp16
    input_len = args.features_num
    if(args.float16):
        input_len = args.features_num * 2

    model = SAGE(in_feats=input_len,
                        n_hidden=args.hidden_dim,
                        n_classes=args.class_num,
                        n_layers=args.hops_num,
                        activation=Func.relu,
                        dropout=args.drop_rate).to(cuda_device)

    if dist.is_initialized():
        model = DDP(model, device_ids=[device_id])
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    model.train()

    epoch_num = args.epoch

    for epoch in range(epoch_num):
        forward = 0
        start = time.time()
        epoch_time = 0
        global total_train_time
        global train_batches
        total_train_time = 0
        train_batches = 0
        #print("Timestamp {:d}, sample {:d}, worker {:d}, system {:s}, {:s}".format(time.time_ns(), 0, rank, "sampler","start"))
        for it, (input_nodes, output_nodes, blocks) in enumerate(
            train_dataloader
        ):
            if it >= train_step:
                break
            #print("Timestamp {:d}, sample {:d}, worker {:d}, system {:s}, {:s}".format(time.time_ns(), it, rank, "sampler", "gensample"))
            #print(output_nodes)
            train_loss = train_one_step(model, optimizer, loss_fcn, cuda_device, feat_len, blocks, args.float16, device_id, it)
            #print("Timestamp {:d}, sample {:d}, worker {:d}, system {:s}, {:s}".format(time.time_ns(), it + 1, rank, "sampler","start"))
            #if iter % 100 == 0 and iter != 0:
            #    print('Iter {} Batches: {}, avg train time : {} us'.format(iter, train_batches, total_train_time / train_batches / 1000))
            # if device_id == 0:
            #     print('Iter {} Train Loss :{} '.format(iter, train_loss))

        #print('Batches: {}, avg train time : {} us'.format(train_batches, total_train_time / train_batches / 1000))
        epoch_time += time.time() - start
        
        model.eval()
        metric = torchmetrics.Accuracy('multiclass', num_classes = args.class_num)
        metric = metric.to(device_id)
        model.metric = metric
        with torch.no_grad():
            total_train_time = 0
            train_batches = 0
            for it, (input_nodes, output_nodes, blocks) in enumerate(
                val_dataloader
            ):
                valid_one_step(model, metric, cuda_device, feat_len, blocks, args.float16)
                if it >= valid_step:
                    break
                #if iter % 100 == 0 and iter != 0:
                #    print('Iter {} Batches: {}, avg valid time : {} us'.format(iter, train_batches, total_train_time / train_batches / 1000))
            # if device_id == 0:
            #     print('Iter {} Train Loss :{} '.format(iter, train_loss))

            #print('Iter {} Batches: {}, avg valid time : {} us'.format(iter, train_batches, total_train_time / train_batches / 1000))
            #epoch_time += time.time() - start
            acc_val = metric.compute()
        if device_id == 0:
            print("Epoch:{}, Cost:{} s, Val Acc: {}".format(epoch, epoch_time, acc_val))

    
    model.eval()
    metric = torchmetrics.Accuracy('multiclass', num_classes = args.class_num)
    metric = metric.to(device_id)
    model.metric = metric
    with torch.no_grad():
        for it, (input_nodes, output_nodes, blocks) in enumerate(
            test_dataloader
        ):
            if it >= test_step:
                break
            test_one_step(model, metric, cuda_device, feat_len, blocks, args.float16)
        acc = metric.compute()
    if device_id == 0:
        print("Accuracy on test data: {}".format(acc))
    metric.reset()
    cleanup()

def run_distribute(dist_fn, world_size, args, graph_data):
    print(graph_data[0])
    mp.spawn(dist_fn,
             args=(world_size, args, graph_data),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    cur_path = sys.path[0]
    argparser = argparse.ArgumentParser("Train GNN.")
    argparser.add_argument('--class_num', type=int, default=172)
    argparser.add_argument('--features_num', type=int, default=128)
    argparser.add_argument('--hidden_dim', type=int, default=256)
    argparser.add_argument('--hops_num', type=int, default=2)
    argparser.add_argument('--nbrs_num', type=list, default=[10, 25])
    argparser.add_argument('--drop_rate', type=float, default=0.5)
    argparser.add_argument('--learning_rate', type=float, default=0.003)
    argparser.add_argument('--epoch', type=int, default=2)
    argparser.add_argument('--gpu_number', type=int, default=2)
    argparser.add_argument('--float16', type=str, default=False)
    argparser.add_argument('--dataset_path', type=str, default="./dataset")
    argparser.add_argument('--dataset_name', type=str, default="ukunion")
    argparser.add_argument('--train_batch_size', type=int, default=8000)
    argparser.add_argument('--cache_memory', type=int, default=38000000)
    argparser.add_argument('--compress', type=int, default=0)
    args = argparser.parse_args()
    if args.float16 == "True":
        args.float16 = True
    else:
        args.float16 = False
    world_size = args.gpu_number

    print(args.nbrs_num)

    import load_graph
    g, features, labels, train_idx, val_idx, test_idx = \
        load_graph.load(args.dataset_path, args.dataset_name)
    train_idx_split = []
    val_idx_split = []
    test_idx_split = []
    # Splitting followed by Legion's data loader
    for rank in range(world_size):
        mask = (train_idx % world_size == rank)
        train_idx_split.append(train_idx[mask])

        mask = (val_idx % world_size == rank)
        val_idx_split.append(val_idx[mask])

        mask = (test_idx % world_size == rank)
        test_idx_split.append(test_idx[mask])

    min_train_size = min([len(x) for x in train_idx_split])
    train_step = (min_train_size - 1) // args.train_batch_size

    max_valid_size = max([len(x) for x in val_idx_split])
    valid_step = (max_valid_size - 1) // 512 + 1

    max_test_size = max([len(x) for x in test_idx_split])
    test_step = (max_test_size - 1) // 512 + 1

    print("Train step: {}, Valid step: {}, Test step: {}".format(train_step, valid_step, test_step))

    if(args.compress):
        print("Compressing features")
        g.get_node_storage("feat").compress()

    graph_data = g, train_idx_split, val_idx_split, test_idx_split, \
        train_step, valid_step, test_step
    #TODO: Divide among executing GPUs
    run_distribute(worker_process, world_size, args, graph_data)