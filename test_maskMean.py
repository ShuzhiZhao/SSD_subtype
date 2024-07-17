import torch
import torch.nn as nn
import torch.nn.functional as F

targets = torch.tensor([1,2,0])  # labels
preds = torch.tensor([[1.4,0.5,1.1],[0.7,0.4,0.2],[2.4,0.2,1.4]]) # data

def pad_loss(pred,target,pad_index=None):
    if pad_index == None :  # no padding mask
        mask = torch.ones_like(target,dtype=torch.float)
    else:
        mask = (target != pad_index).float()
    print('mask value:',mask)
    nopd = mask.sum().item()
    # one-hot code
    target = torch.zeros(pred.shape).scatter(dim=1,index=target.unsqueeze(-1),source=torch.tensor(1))
    target_ = target*mask.unsqueeze(-1) # mask
    print('target_:',target_)
    loss = -(F.log_softmax(pred,dim=-1)*target_.float()).sum()/nopd # NLL loss
    return loss

loss1 = pad_loss(preds,targets)
print('loss1:',loss1)

loss2 = pad_loss(preds,targets,pad_index=0)
print('loss2:',loss2)