import torch
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv,SAGEConv
from torch_geometric.datasets import Planetoid

def encode_onehot(labels):                                   # 把标签转换成onehot
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def normalize(mx):                                           # 归一化
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def  load_data(path="./", dataset="cora"):
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),# 读取节点特征和标签
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32) # 读取节点特征
    dict = {int(element):i for i,element in enumerate(idx_features_labels[:, 0:1].reshape(-1))}    # 建立字典
    labels = encode_onehot(idx_features_labels[:, -1])                       # 标签用onehot方式表示
    e = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)    # 读取边信息
    edges = []
    for i, x in enumerate(e):
        edges.append([dict[e[i][0]], dict[e[i][1]]])                         # 若A->B有变 则B->A 也有边                  
        edges.append([dict[e[i][1]], dict[e[i][0]]])                         # 给的数据是没有从0开始需要转换
    features = normalize(features)                                           # 特征值归一化       
    features = torch.tensor(np.array(features.todense()), dtype=torch.float32)
    labels = torch.LongTensor(np.where(labels)[1])
    edges = torch.tensor(edges, dtype=torch.int64).T
    return features, edges, labels

class GAT(torch.nn.Module):
    def __init__(self, feature, hidden, classes, heads=1):
        super(GAT,self).__init__()
        self.gat1 = GATConv(feature, hidden, heads=heads)
        self.gat2 = GATConv(hidden*heads, classes)
    def forward(self, features, edges):
        features = self.gat1(features, edges)       # edges 这里输入是(1,2),表示1和2有边相连。
        features = F.relu(features)
        features = F.dropout(features, training=self.training)
        features = self.gat2(features, edges)
        return F.log_softmax(features, dim=1)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = GAT(1433, 8, 7,heads=4).to(device)
# features, edges, labels = load_data()
# idx_train = range(2000)                              
# idx_test = range(2000, 2700)
# idx_train = torch.LongTensor(idx_train)
# idx_test = torch.LongTensor(idx_test)
# optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
# model.train()
# for epoch in range(200):
#     optimizer.zero_grad()
#     out = model(features.to(device), edges.to(device))
#     loss = F.nll_loss(out[idx_train], labels[idx_train].to(device))
#     loss.backward()
#     optimizer.step()
#     print(f"epoch:{epoch+1}, loss:{loss.item()}")        
