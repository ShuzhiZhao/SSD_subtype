import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from info_nce import InfoNCE,info_nce
from Encoder import DeepWalk,node2vec,Graph2vec,GCN,GAT,GraphRNN

class HydraNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.resnet18(pretrained=True)
        self.n_features = self.net.fc.in_features
        self.net.fc = nn.Identify()
        
        self.net.fc1 = nn.Sequential(OrderedDict([('linear',nn.Linear(self.n_features,self.n_features)),('relu1',nn.ReLU()),('final',nn.Linear(self.n_feaures,1))]))
        self.net.fc2 = nn.Sequential(OrderedDict([('linear',nn.Linear(self.n_features,self.n_features)),('relu1',nn.ReLU()),('final',nn.Linear(self.n_feaures,1))]))
        
    def forward(self,x):
        cla_head = self.net.fc1(self.net(x))
        reg_head = self.net.fc2(self.net(x))
        
        return cla_head,reg_head
    
class MFCLM(nn.Module):
    def __init__(self):
        super(MFCLM,self).__init__()
        Sel = 'CNNRNN'
        if 'CNNRNN' in Sel:
            self.encoder1 = nn.Conv1d(62,62,3,stride=2) # DeepWalk,node2vec,Graph2vec
            self.encoder2 = nn.RNN(62,62,2) # GCN,GAT,GraphRNN    
        elif 'NodeEdge' in Sel:
            # DeepWalk,node2vec,Graph2vec
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=62,nhead=8) 
            self.encoder1 = nn.TransformerEncoder(encoder_layer,num_layers=6) 
            # DeepWalk,node2vec,Graph2vec
            self.encoder2 = nn.Embedding(10,62)
#         self.Encoder1 = models.resnet18(pretrained=True) 
#         self.Encoder2 = models.resnet18(pretrained=True)     
        
    def forward(self,x):
        loss = InfoNCE()
        FT1 = self.encoder1(x)
        FT1 = FT1.reshape(FT1.shape[0],-1)
        M1 = nn.Linear(FT1.shape[1],3844)
        FT1 = M1(FT1)
        FT2 = self.encoder2(x)
        FT2 = FT2[0].reshape(FT2[0].shape[0],-1)
        M2 = nn.Linear(FT2.shape[1],3844)
        FT2 = M2(FT2)
        FT = FT1*FT2
        FT = F.normalize(FT,dim=0)
#         NCE = loss(torch.Tensor(FT1.reshape(FT1.shape[0],62,62)),torch.Tensor(FT2.reshape(FT2.shape[0],62,62)))
        print('FT shape:',FT.shape)

        return FT
        