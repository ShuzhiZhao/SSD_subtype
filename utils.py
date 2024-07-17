import torch
import torch.nn as nn
from torch.utils.data import Dataset

class mulTaskData(Dataset):
    def __init__(self,data_fre1,data_fre2,subtype,pre):
#         self.data_paths = data_paths
        self.data_fre1 = data_fre1
        self.data_fre2 = data_fre2
        self.clas = subtype
        self.regs = pre
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,index):
        data_fre1 = self.data_fre1[index]
        data_fre2 = self.data_fre2[index]
        cla = self.clas[index]
        reg = self.regs[index]
        
        sample = {'data_fre1':data_fre1,'data_fre2':data_fre2,'class':cla,'regression':reg}
        return sample
        
class SimCLR(nn.Module):
    def __init__(
        self,
        net,
        image_size,
        channels = 3,
        hidden_layer = -2,
        project_hidden = True,
        project_dim = 128,
        augment_both = True,
        use_nt_xent_loss = False,
        augment_fn = None,
        temperature = 0.1
    ):
        super().__init__()
        self.net = NetWrapper(net, project_dim, layer = hidden_layer)
        self.augment = default(augment_fn, get_default_aug(image_size, channels))
        self.augment_both = augment_both
        self.temperature = temperature

        # get device of network and make wrapper same device
        device = get_module_device(net)
        self.to(device)

        # send a mock image tensor to instantiate parameters
        self.forward(torch.randn(1, channels, image_size, image_size))

    def forward(self, x):
        b, c, h, w, device = *x.shape, x.device
        transform_fn = self.augment if self.augment_both else noop
        # 把原图使用不同数据增强和ViT提取成两个不同的图像特征(正样本对queries、keys)
        queries, _ = self.net(transform_fn(x))  
        keys, _    = self.net(self.augment(x))

        queries, keys = map(flatten, (queries, keys))
        # 计算loss
        loss = nt_xent_loss(queries, keys, temperature = self.temperature) 
        return loss
    
def nt_xent_loss(queries, keys, temperature = 0.1):
    b, device = queries.shape[0], queries.device

    n = b * 2  # 同一图片内部不同patch也是负样本
    projs = torch.cat((queries, keys))
    logits = projs @ projs.t()

    mask = torch.eye(n, device=device).bool()
    logits = logits[~mask].reshape(n, n - 1)  # 同一图片内部不同patch也是负样本，除了自己和自己
    logits /= temperature

    labels = torch.cat(((torch.arange(b, device = device) + b - 1), torch.arange(b, device=device)), dim=0)
    loss = F.cross_entropy(logits, labels, reduction = 'sum')
    loss /= n
    return loss
    
        