import math
import time
import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

class GraphConvolution(Module):                            
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):             # 这里代码做了简化如 3.2节。
        support = torch.mm(input, self.weight) # (2708, 16) = (2708, 1433) X (1433, 16)
        output = torch.spmm(adj, support)      # (2708, 16) = (2708, 2708) X (2708, 16)
        if self.bias is not None:
            return output + self.bias          # 加上偏置 (2708, 16)
        else:
            return output                      # (2708, 16)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):                                             # 定义两层GCN
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = torch.nn.functional.relu(self.gc1(x, adj))
        x = torch.nn.functional.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return torch.nn.functional.log_softmax(x, dim=1)       # 对每一个节点做softmax

#         model = GCN(nfeat=features.shape[1], nhid=hidden,nclass=labels.max().item() + 1,dropout=dropout)

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()                                                            # 梯度清零
    output = model(features, adj)                         
    loss_train = torch.nn.functional.nll_loss(output[idx_train], labels[idx_train])  # 损失函数
    acc_train = accuracy(output[idx_train], labels[idx_train])                       # 计算准确率
    loss_train.backward()                                                            # 反向传播
    optimizer.step()                                                                 # 更新梯度

    if not fastmode:
        model.eval()
        output = model(features, adj)

    loss_val = torch.nn.functional.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    
def test():
    model.eval()
    output = model(features, adj)                     # features:(2708, 1433)   adj:(2708, 2708)
    loss_test = torch.nn.functional.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden = 16                                     # 定义隐藏层数
dropout = 0.5
lr = 0.01 
weight_decay = 5e-4
fastmode = 'store_true'

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def  load_data(path="./", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),# 读取节点标签
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32) # 读取节点特征
    labels = encode_onehot(idx_features_labels[:, -1])                       # 标签用onehot方式表示
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)                
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),      # 读取边信息
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    features = normalize(features)                                            # 特征值归一化          
    adj = normalize(adj + sp.eye(adj.shape[0]))                               # 边信息归一化

    idx_train = range(140)                                                    # 训练集
    idx_val = range(200, 500)                                                 # 验证集
    idx_test = range(500, 1500)                                               # 测试集

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)                               # 转换成邻居矩阵

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
                                                           
    return adj, features, labels, idx_train, idx_val, idx_test            
# adj, features, labels, idx_train, idx_val, idx_test = load_data()

# optimizer = torch.optim.Adam(model.parameters(),lr=lr, weight_decay=weight_decay)
# if device:                                          # 数据放在cuda上
#     model.cuda()
#     features = features.cuda()
#     adj = adj.cuda()
#     labels = labels.cuda()
#     idx_train = idx_train.cuda()
#     idx_val = idx_val.cuda()
#     idx_test = idx_test.cuda()
# epochs = 500
# for epoch in range(epochs):
#     train(epoch)
# test()    
