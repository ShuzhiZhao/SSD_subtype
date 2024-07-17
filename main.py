import os
import scipy.io as scio
import numpy as np
import torch.nn as nn
from Models import HydraNet,MFCLM
from utils import mulTaskData
from torch.utils.data import DataLoader
from sklearn.model_selection import cross_val_score, train_test_split,ShuffleSplit
from info_nce import InfoNCE,info_nce
from cluster_index import CH_EvaInd_Vis
import torch
from sklearn import manifold
import torch.nn.functional as F
import pandas as pd
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time

## load Data of NS
def loadData(data_dir):
    feature = ['psd','coh','wpli','powenv']
    freq = ['delta','theta','alpha','beta1','beta2','gamma1','gamma2']
    # double frequency selection
    DFS = []
    data = []
    files_sub = os.listdir(data_dir)
    sub_num = 0
    loss = InfoNCE()
    pd.set_option('display.unicode.ambiguous_as_wide',True)
    pd.set_option('display.unicode.east_asian_width',True)    
    info_path = data_dir.split(data_dir.split('/')[-2])[0]+'info.xlsx'
    df = pd.read_excel(r''+info_path,sheet_name='Sheet2',usecols=['数据名称','总分'])
    print(df)
    from multi_process import MultiProcess
    MP = MultiProcess()
    for FQ1 in range(len(freq)):
        for FQ2 in range(len(freq)):
            if FQ1>FQ2 :
                DFS.append([FQ1,FQ2])
    print('+++++++++++++++++++ load data ... ++++++++++++++++++') 
    for dfs in DFS:
        if ('beta' in freq[dfs[0]] and 'beta' in freq[dfs[1]]) or ('gamma' in freq[dfs[0]] and 'gamma' in freq[dfs[1]]) :
            print('The same frequency')
        else:    
            print('!!!!!!!!!!!!@@@@@@@@@@ double frequency selection:',freq[dfs[0]],freq[dfs[1]]) 
            SpuLab_ = []
            data_ = []
            Score_ = []
        # load data(EC|EO*3Net)
        for subs in files_sub:
            files = os.listdir(data_dir+'/'+subs)
            # load info.xlsx files
            keyName = subs.split('_'+subs.split('_')[-1])[0]
            df_index = df[df['数据名称'].str.contains(keyName,case=False)].index.tolist()
#             print('keyName:',keyName,df['数据名称'].str.contains(keyName,case=False).any())
#             print('index:',df_index,'neuro-11 value:',df['总分'][df_index[0]])
            score_neuro11 = df['总分'][df_index[0]]
            for file in files:
#                 print('sub:',subs,'\nfile:',file)
                samplesfile = scio.loadmat(data_dir+'/'+subs+'/'+file) 
                keywords = list((samplesfile.keys()))
                if '_channel_' in file :
#                     print('++++File',file,'++++Keywords',keywords)
                    for i in keywords:
                        if 'Features' in i:
                            samples = samplesfile[i]
                            for FT in range(len(feature)):
                                if 'psd' not in feature[FT]:
                                    sub_num = sub_num+1
                                    data_,SpuLab_,Score_ = doubleFreCL(FT,dfs,data_,SpuLab_,Score_,samples,loss,score_neuro11)
#                                     temp = (FT,dfs,data_,SpuLab_,Score_,samples,loss,score_neuro11)
#                                     data_,SpuLab_,Score_ = MP.multi_with_result(func=doubleFreCL,arg_list=temp, process_num=6)
        tsne = manifold.TSNE(n_components=1,init='pca',random_state=501)
        Y_tsne = tsne.fit_transform(np.array(SpuLab_)) 
        temp = np.mean(Y_tsne)
        Y_tsne[Y_tsne<temp] = 0
        Y_tsne[Y_tsne>=temp] = 1
        model = MFCLM()        
        print('Label:',Y_tsne.shape,'data shape:',np.array(data_).shape,'Score shape:',np.array(Score_).shape)
        SelfSupervisedLearning(model,data_,Y_tsne,Score_,freq[dfs[0]]+'_'+freq[dfs[1]])
#         print('!!!!!!!!!!!!!!! data with double prequency shape:',np.array(data).shape)  
    print('sub_num:',sub_num)
    return data

def doubleFreCL(FT,dfs,data_,SpuLab_,Score_,samples,loss,score_neuro11):
    fre1 = list(samples[0][0])[FT][0][0][dfs[0]]
    fre2 = list(samples[0][0])[FT][0][0][dfs[1]]
    fre1 = np.array(fre1)
    fre1[np.isnan(fre1)] = 0
    fre2 = np.array(fre2)
    fre2[np.isnan(fre2)] = 0
    l1B = loss(torch.Tensor(fre1),torch.Tensor(fre2))
    l2 = loss(torch.Tensor(fre1),torch.inverse(torch.Tensor(fre1)))
    l3 = loss(torch.Tensor(fre2),torch.inverse(torch.Tensor(fre2)))
#     print('fre1:',fre1,'fre2:',fre2)
#     print('l1B:',l1B,'l2:',l2,'l3:',l3)
    SpuLab = [l1B.item(),l2.item(),l3.item()]
#     print('MF-CL spurious label:',SpuLab)
    SpuLab_.append(SpuLab)
    temp = torch.Tensor((fre1*fre2).tolist())
    temp = F.normalize(temp).numpy().tolist()
    if len(np.argwhere(np.isnan(temp))) == 1:
        print('data_ exit nan',np.argwhere(np.isnan(temp)))
    data_.append(temp)
    Score_.append(score_neuro11)
    
    return data_,SpuLab_,Score_

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.dropout(F.relu(self.hidden(x)))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x 

## train and test set
def crossDataLabel(FT,labels):
    ## 5 cross valition
    test_size = 0.2
    randomseed=1234
    test_sub_num = len(FT)
    print('test_sub_num: ',test_sub_num)
    rs = np.random.RandomState(randomseed)
    train_sid, test_sid = train_test_split(range(test_sub_num), test_size=test_size, random_state=rs, shuffle=True)
    print('training on %d subjects, validating on %d subjects' % (len(train_sid), len(test_sid)))
    ####train set 
    fmri_data_train = [FT[i] for i in train_sid]
    trainLabels = pd.DataFrame(np.array([labels[i] for i in train_sid]))
#     print(type(trainLabels),'\n',trainLabels)
    ERP_train_dataset = ERP_matrix_datasets(fmri_data_train, trainLabels, isTrain='train')
    trainData = DataLoader(ERP_train_dataset)

    ####test set
    fmri_data_test = [FT[i] for i in test_sid]
    testLabels = pd.DataFrame(np.array([labels[i] for i in test_sid]))
#     print(type(testLabels),'\n',testLabels)
    ERP_test_dataset = ERP_matrix_datasets(fmri_data_test, testLabels, isTrain='test')
    testData = DataLoader(ERP_test_dataset)
    
    return trainData,trainLabels,testData,testLabels    

##load data from array
class ERP_matrix_datasets(Dataset):
    ##build a new class for own dataset
    import numpy as np
    def __init__(self, fmri_data_matrix, label_matrix,
                 isTrain='train', transform=False):
        super(ERP_matrix_datasets, self).__init__()

        if not isinstance(fmri_data_matrix, np.ndarray):
            self.fmri_data_matrix = np.array(fmri_data_matrix)
        else:
            self.fmri_data_matrix = fmri_data_matrix
        
        self.Subject_Num = self.fmri_data_matrix.shape[0]
        self.Region_Num = self.fmri_data_matrix[0].shape[-1]

        if isinstance(label_matrix, pd.DataFrame):
            self.label_matrix = label_matrix
        elif isinstance(label_matrix, np.ndarray):
            self.label_matrix = pd.DataFrame(data=np.array(label_matrix))

        self.data_type = isTrain
        self.transform = transform

    def __len__(self):
        return self.Subject_Num

    def __getitem__(self, idx):
        #step1: get one subject data
        fmri_trial_data = self.fmri_data_matrix[idx]
#         fmri_trial_data = fmri_trial_data.reshape(1,fmri_trial_data.shape[0])
        label_trial_data = np.array(self.label_matrix.iloc[idx])
#         print('fmri_trial_data\n{}\n======\nlabel_trial_data\n{}\n'.format(fmri_trial_data.shape,label_trial_data.shape))
        tensor_x = torch.stack([torch.FloatTensor(fmri_trial_data[ii]) for ii in range(len(fmri_trial_data))])  # transform to torch tensors
        tensor_y = torch.stack([torch.LongTensor([int(label_trial_data[ii])]) for ii in range(len(label_trial_data))])
#         print('tensor_x\n{}\n=======\ntensor_y\n{}\n'.format(tensor_x.size(),tensor_y.size()))
        return tensor_x, tensor_y

def SelfSupervisedLearning(model,data,label,score,douFre):
    result = {}
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    trainData,trainLabels,testData,testLabels = crossDataLabel(data,label)
    x = model(torch.FloatTensor(data))
    result_dir = "/media/lhj/Momery/Microstate_HJL/NS_subtype/Result/cluster1/"+douFre
    CH_EvaInd_Vis(x,label,result_dir)
    net = Net(n_feature=62,n_hidden=62,n_output=1).to(device)
    optimizer = optim.Adam(net.parameters(),lr=0.001, weight_decay=5e-4)
    loss_func = nn.CrossEntropyLoss()
    num_epochs=15
    
    model_fit_evaluate(net,device,trainData,trainLabels,testData,testLabels,optimizer,loss_func,num_epochs)
#     print("{} paramters to be trained in the model\n".format(count_parameters(net)))
    
    return result

## train and test model
def model_fit_evaluate(model,device,trainData,trainLabels,testData,testLabels,optimizer,loss_func,num_epochs=100):
    best_acc = 0 
    model_history={}
    model_history['train_loss']=[];
    model_history['train_acc']=[];
    model_history['test_loss']=[];
    model_history['test_acc']=[]; 
    model_history['test_cow_']=[];
    model_history['test_col_']=[];
    model_history['test_value_']=[]; 
    for epoch in range(num_epochs):
        train_loss,train_acc =train(model,device,trainData,trainLabels,optimizer,loss_func,epoch)
        model_history['train_loss'].append(train_loss)
        model_history['train_acc'].append(train_acc)

        test_loss,test_acc = test(model,device,testData,testLabels,loss_func)
        model_history['test_loss'].append(test_loss)
        model_history['test_acc'].append(test_acc)        
        if test_acc > best_acc:
            best_acc = test_acc
            print("Model updated: Best-Acc = {:4f}".format(best_acc))
    for ii in range(20):
        torch.cuda.empty_cache()
    print("best testing accuarcy:",best_acc)
#     plot_history(model_history)
    
##training the model
def train(model, device,train_loader, trainLabels, optimizer,loss_func, epoch):
    model.train()

    acc = 0.
    train_loss = 0.
    total = 0
    t0 = time.time()
    Predict_Scores = []
    True_Scores = []
    for batch_idx, (data,target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        data = data.squeeze(0)
        target = target.view(-1).float().random_(62)
#         print('inputs ',data.size(),'labels:',target.size())
#         print('inputs ',data,'labels:',target)
        out = model(data)
        Predict_Scores.append(out),True_Scores.append(target)
        loss = loss_func(out.t().float(),target.long())
        pred = F.log_softmax(out, dim=1).argmax(dim=1)[0]

        total += target.size(0)
        train_loss += loss.sum().item()
        acc += pred.eq(target.view_as(pred)).sum().item()
        
        loss.backward()
        optimizer.step()
        
        # other ways Scatter_
#         test_loss += loss.item()
#         _,predicted = outputs.max(1)
#         pre_mask = torch.zeros(outputs.size()).scatter_(1,predicted.to(device).view(-1,1),1.)
#         predict_num += pre_mask.sum(0)
#         tar_mask = torch.zeros(outputs.size()).scatter_(1,target.data.to(device).view(-1,1),1.)
#         target_num += tar_mask.sum(0)
#         acc_mask = pre_mask*tar_mask
#         acc_num += acc_mask.sum(0)
#     recall = acc_num/target_num
#     precision =acc_num/predict_num
#     F1 = 2*recall*precision/(recall+precision)
#     accuracy = 100.*acc_num.sum(1)/target_num.sum(1)
        
    print("\nEpoch {}: \nTime Usage:{:4f} | Training Loss {:4f} | Acc {:4f}".format(epoch,time.time()-t0,train_loss/total,acc/total))
    return train_loss/total,acc/total

def test(model, device, test_loader, testLabels, loss_func):
    model.eval()
    test_loss=0.
    test_acc = 0.
    total = 0
    ##no gradient desend for testing
    with torch.no_grad():
        Predict_Scores = []
        True_Scores = []
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            out = model(data)
#             print('input:',target,'predict:',out.tolist()[0][0][0])
            Predict_Scores.append(out.tolist()[0][0][0]+np.random.uniform(-2,2)),True_Scores.append(target.tolist()[0][0][0]+np.random.uniform(-2,2))
            out = out.reshape((out.shape[0],-1))
            target=target.reshape((-1))
            loss = loss_func(out.float(),target.random_(62).long())
            test_loss += loss.sum().item()
            pred = F.log_softmax(out, dim=1).argmax(dim=1)
            #pred = out.argmax(dim=1,keepdim=True) # get the index of the max log-probability
            total += target.size(0)
            test_acc += pred.eq(target.view_as(pred)).sum().item()                        
    
    test_loss /= total
    test_acc /= total
    print('Test Loss {:4f} | Acc {:4f}'.format(test_loss,test_acc))
    return test_loss,test_acc

## main function
def Main(work_dir):
    # data and label
    data = loadData(work_dir+'/Data/')
    
    
#     train_dataloader = DataLoader(mulTaskData(train_dataset),shuffle=True,batch_size=BATCH_SIZE)
#     val_dataloader = DataLoader(mulTaskData(valid_dataset),shuffle=True,batch_size=BATCH_SIZE)
    
#     # multi-task with self-supervised of subtype and predict PSD
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     model = HydraNet().to(device=device)
#     cla_loss = nn.CrossEntropyLoss()
#     reg_loss = nn.L1Loss()
#     sig = nn.Sigmoid()
#     optimizer = torch.optim.SGD(model.parameters(),lr=1e-4,momentum=0.09)
    

    
work_dir = "/media/lhj/Seagate Basic/SSD"
Main(work_dir)