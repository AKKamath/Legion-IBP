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

import ipc_service
import dgl
from dgl.nn.pytorch import SAGEConv
from dgl.heterograph import DGLBlock
import time
import numpy as np
import torchmetrics
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
def train_one_step(model, optimizer, loss_fcn, device, feat_len, iter, device_id, fp16):
    
    ids, features, labels, block1_agg_src, block1_agg_dst, block2_agg_src, block2_agg_dst = ipc_service.get_next(feat_len)
    block1_src_num, block1_dst_num, block2_src_num, block2_dst_num = ipc_service.get_block_size()

    data_type = torch.float32
    # Typecast fp32 to fp16
    if fp16:
        features = features.view(torch.float16)
        data_type = torch.float16
    blocks = []
    blocks.append(create_dgl_block(block1_agg_src, block1_agg_dst, block1_src_num, block1_dst_num, fp16))
    blocks.append(create_dgl_block(block2_agg_src, block2_agg_dst, block2_src_num, block2_dst_num, fp16))
    # print(features[:100])
    # print(ids[:100])
    start = time.perf_counter_ns()
    with autocast('cuda', enabled=fp16, dtype=data_type):
        #print("Timestamp {:d}, sample {:d}, worker {:d}, system {:s}, {:s}".format(time.time_ns(), iter, device_id, "trainer", "getsample"))
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
    #print("Timestamp {:d}, sample {:d}, worker {:d}, system {:s}, {:s}".format(time.time_ns(), iter, device_id, "trainer", "finishtrain"))
    global total_train_time
    global train_batches
    total_train_time += end - start
    train_batches += 1
    ipc_service.synchronize()
    return 0

def valid_one_step(model, metric, device, feat_len, fp16):
    
    ids, features, labels, block1_agg_src, block1_agg_dst, block2_agg_src, block2_agg_dst = ipc_service.get_next(feat_len)
    block1_src_num, block1_dst_num, block2_src_num, block2_dst_num = ipc_service.get_block_size()

    data_type = torch.float32
    # Typecast fp32 to fp16
    if fp16:
        features = features.view(torch.float16)
        data_type = torch.float16
    blocks = []
    blocks.append(create_dgl_block(block1_agg_src, block1_agg_dst, block1_src_num, block1_dst_num, fp16))
    blocks.append(create_dgl_block(block2_agg_src, block2_agg_dst, block2_src_num, block2_dst_num, fp16))
    start = time.perf_counter_ns()
    with autocast('cuda', enabled=fp16, dtype=data_type):
        batch_pred = model(blocks, features)
    long_labels = torch.as_tensor(labels, dtype=torch.long, device=device)
    batch_pred = torch.softmax(batch_pred, dim=1).to(device)
    acc = metric(batch_pred, long_labels)
    ipc_service.synchronize()
    end = time.perf_counter_ns()
    global total_train_time
    global train_batches
    total_train_time += end - start
    train_batches += 1
    return acc

def test_one_step(model, metric, device, feat_len, fp16):
    
    ids, features, labels, block1_agg_src, block1_agg_dst, block2_agg_src, block2_agg_dst = ipc_service.get_next(feat_len)
    block1_src_num, block1_dst_num, block2_src_num, block2_dst_num = ipc_service.get_block_size()
    data_type = torch.float32
    # Typecast fp32 to fp16
    if fp16:
        features = features.view(torch.float16)
        data_type = torch.float16
    blocks = []
    blocks.append(create_dgl_block(block1_agg_src, block1_agg_dst, block1_src_num, block1_dst_num, fp16))
    blocks.append(create_dgl_block(block2_agg_src, block2_agg_dst, block2_src_num, block2_dst_num, fp16))
    with autocast('cuda', enabled=fp16, dtype=data_type):
        batch_pred = model(blocks, features)
        long_labels = torch.as_tensor(labels, dtype=torch.long, device=device)
        batch_pred = torch.softmax(batch_pred, dim=1).to(device)
    acc = metric(batch_pred, long_labels)
    ipc_service.synchronize()
    return acc

def worker_process(rank, world_size, args):
    print(f"Running GNN Training on CUDA {rank}.")
    device_id = rank
    setup(rank, world_size)
    cuda_device = torch.device("cuda:{}".format(device_id))
    torch.cuda.set_device(cuda_device)
    ipc_service.initialize()
    train_steps, valid_steps, test_steps = ipc_service.get_steps()

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
        for iter in range(train_steps):
            train_loss = train_one_step(model, optimizer, loss_fcn, cuda_device, feat_len, iter, device_id, args.float16)
            #if iter % 100 == 0 and iter != 0:
            #    print('Iter {} Batches: {}, avg train time : {} us'.format(iter, train_batches, total_train_time / train_batches / 1000))
            # if device_id == 0:
            #     print('Iter {} Train Loss :{} '.format(iter, train_loss))

        print('Iter {} Batches: {}, avg train time : {} us'.format(iter, train_batches, total_train_time / train_batches / 1000))
        epoch_time += time.time() - start
        
        model.eval()
        metric = torchmetrics.Accuracy('multiclass', num_classes = args.class_num)
        metric = metric.to(device_id)
        model.metric = metric
        with torch.no_grad():
            total_train_time = 0
            train_batches = 0
            for iter in range(valid_steps):
                valid_one_step(model, metric, cuda_device, feat_len, args.float16)
                #if iter % 100 == 0 and iter != 0:
                #    print('Iter {} Batches: {}, avg valid time : {} us'.format(iter, train_batches, total_train_time / train_batches / 1000))
            # if device_id == 0:
            #     print('Iter {} Train Loss :{} '.format(iter, train_loss))

            print('Iter {} Batches: {}, avg valid time : {} us'.format(iter, train_batches, total_train_time / train_batches / 1000))
            epoch_time += time.time() - start
            acc_val = metric.compute()
        if device_id == 0:
            print("Epoch:{}, Cost:{} s, Val Acc: {}".format(epoch, epoch_time, acc_val))

    
    model.eval()
    metric = torchmetrics.Accuracy('multiclass', num_classes = args.class_num)
    metric = metric.to(device_id)
    model.metric = metric
    with torch.no_grad():
        for iter in range(test_steps):
            test_one_step(model, metric, cuda_device, feat_len, args.float16)
        acc = metric.compute()
    if device_id == 0:
        print("Accuracy on test data: {}".format(acc))
    metric.reset()

    ipc_service.finalize()
    cleanup()

def run_distribute(dist_fn, world_size, args):
    mp.spawn(dist_fn,
             args=(world_size, args),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    cur_path = sys.path[0]
    argparser = argparse.ArgumentParser("Train GNN.")
    argparser.add_argument('--class_num', type=int, default=172)
    argparser.add_argument('--features_num', type=int, default=128)
    argparser.add_argument('--hidden_dim', type=int, default=256)
    argparser.add_argument('--hops_num', type=int, default=2)
    argparser.add_argument('--nbrs_num', type=list, default=[25, 10])
    argparser.add_argument('--drop_rate', type=float, default=0.5)
    argparser.add_argument('--learning_rate', type=float, default=0.003)
    argparser.add_argument('--epoch', type=int, default=2)
    argparser.add_argument('--gpu_number', type=int, default=2)
    argparser.add_argument('--float16', type=str, default=False)
    args = argparser.parse_args()
    if args.float16 == "True":
        args.float16 = True
    else:
        args.float16 = False
    world_size = args.gpu_number

    run_distribute(worker_process, world_size, args)
