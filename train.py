import os
import time
import pickle
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import seaborn as sns
from scipy import interpolate
from datetime import datetime
import pandas as pd
from tool import EarlyStopping
from sklearn.metrics import roc_auc_score,mean_squared_error,mean_absolute_error, r2_score
from common import *
from net import CRNN

import torch
import torch.nn.functional as F
import torch.optim as optim

from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, Sampler, TensorDataset
from torch.utils.data.sampler import RandomSampler

from config import config

hairlab_data = config.hairlab_data

# 解决中文和负号显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import warnings
warnings.filterwarnings('ignore')



### data preparation ###

n_cyc = 30          # 循环数
in_stride = 3       # 间隔数
fea_num = 100       # 插值特征数

v_low = 3.36        # 电压最小值
v_upp = 3.60        # 电压最大值
q_low = 610         # 容量最小值 mAh
q_upp = 1190        # 容量最大值
rul_factor = 3000   # 寿命归一化因子 （假设最大寿命循环数）
cap_factor = 1190   # 容量归一化因子


# pkl_list = os.listdir(hairlab_data)
# pkl_list = sorted(pkl_list, key=lambda x:int(x.split('-')[0])*10 + int(x[-5]))

# train_name = []
# for name in pkl_list:
#     train_name.append(name[:-4])

# all_loader = dict()
# print('----init_train----')
# for name in train_name:
#     tmp_fea, tmp_lbl = get_xy(name, n_cyc, in_stride, fea_num, v_low, v_upp, q_low, q_upp, rul_factor, cap_factor)
#     all_loader.update({name:{'fea':tmp_fea,'lbl':tmp_lbl}})

# loader_path = os.path.join('D:/wzt/bishe_data/data/HUST_cycle_dataloader/', 'all_loader.pkl')
# with open(loader_path, 'wb') as f:
#     pickle.dump(all_loader, f)
# print('save over')


loader_path  = 'D:/wzt/bishe_data/data/HUST_cycle_dataloader/all_loader.pkl'
with open(loader_path, 'rb') as file:
    all_loader = pickle.load(file)

### data loader ###

# new_valid = ['4-3', '5-7', '3-3', '2-3', '9-3', '10-5', '3-2', '3-7']
# new_train = ['9-1', '2-2', '4-7','9-7', '1-8','4-6','2-7','8-4', '7-2','10-3', '2-4', '7-4', '3-4',
#             '5-4', '8-7','7-7', '4-4','1-3', '7-1','5-2', '6-4', '9-8','9-5','6-3','10-8','1-6','3-5',
#              '2-6', '3-8', '3-6', '4-8', '7-8','5-1', '2-8', '8-2','1-5','7-3', '10-2','5-5', '9-2','5-6', '1-7', 
#              '8-3', '4-1','4-2','1-4','6-5', ]
# new_test  = ['9-6','4-5','1-2', '10-7','1-1', '6-1','6-6', '9-4','10-4','8-5', '5-3','10-6',
#             '2-5','6-2','3-1','8-8', '8-1','8-6','7-6','6-8','7-5','10-1']

new_valid = ['4-3', '5-7',]
new_train = ['9-1', '2-2', '4-7','9-7', '1-8','4-6', ]
new_test  = ['9-6','4-5','1-2', '10-7',]

stride = 10
train_fea, train_lbl = [], []
for name in new_train + new_valid:
    tmp_fea, tmp_lbl = all_loader[name]['fea'],all_loader[name]['lbl']
    tmp_fea_reversed = tmp_fea[::-1]
    train_fea.append(tmp_fea_reversed[::stride][::-1])     # 指定步长间隔采样
    tmp_lbl_reversed = tmp_lbl[::-1]
    train_lbl.append(tmp_lbl_reversed[::stride][::-1])
train_fea = np.vstack(train_fea)        # len = 55 (10279, 10, 100, 4)  <-- (55, n, 10, 100, 4)
train_lbl = np.vstack(train_lbl).squeeze() # (10279, 11) <-- 

stride = 10
valid_fea, valid_lbl = [], []
for name in new_test:
    tmp_fea, tmp_lbl = all_loader[name]['fea'],all_loader[name]['lbl']
    tmp_fea_reversed = tmp_fea[::-1]
    valid_fea.append(tmp_fea_reversed[::stride][::-1])
    tmp_lbl_reversed = tmp_lbl[::-1]
    valid_lbl.append(tmp_lbl_reversed[::stride][::-1])
valid_fea = np.vstack(valid_fea)
valid_lbl = np.vstack(valid_lbl).squeeze()

# print(train_fea.shape, train_lbl.shape, valid_fea.shape, valid_lbl.shape)

seed_torch(0)
batch_size = 256

train_fea_ = train_fea[:].copy()
train_lbl_ = train_lbl[:].copy()

train_fea_ = train_fea_.transpose(0,3,2,1)      # (10279, 4, 100, 10)
valid_fea_ = valid_fea.transpose(0,3,2,1)       # (4074, 4, 100, 10)

trainset = TensorDataset(torch.Tensor(train_fea_), torch.Tensor(train_lbl_))
validset = TensorDataset(torch.Tensor(valid_fea_), torch.Tensor(valid_lbl))

train_loader = DataLoader(trainset, batch_size=batch_size,shuffle=True)
valid_loader = DataLoader(validset, batch_size=batch_size,)

train_loader_t = DataLoader(trainset, batch_size=batch_size,shuffle=False)


### model pre-training ###

'''
lamda (float): The weight of RUL loss
alpha (List: [float]): The weights of Capacity loss
'''         
lamda = 1e-2
alpha = torch.Tensor([0.1] * 10 )

'''
model training 
'''
tic = time.time()
seed_torch(0)
device = 'cuda'
model = CRNN(100,4,64,64)
model = model.to(device)

num_epochs = 200

trainer = Trainer(lr = 8e-4, n_epochs = num_epochs,device = device, patience = 1200,
                  lamda = lamda, alpha = alpha, model_name='./HAIRLAB/ckpt/HUST/wx_inner_pretrain')
model ,train_loss, valid_loss, total_loss = trainer.train(train_loader, valid_loader, model)

print(f'pretrain_time:{time.time()-tic}')

### online transfer ###

lamda = 0.0
train_weight9 = [0., 0.1, 0., 0., 0.1, 0., 0., 0., 0.,]
valid_weight9 = [0. if (i!=0) else 0.1 for i in train_weight9]
train_alpha = torch.Tensor(train_weight9 + [0.] )
valid_alpha = torch.Tensor(valid_weight9 + [0.])
device = 'cuda'

pretrain_model_path = './HAIRLAB/ckpt/HUST/wx_inner_pretrain_end.pt'
finetune_model_path = './HAIRLAB/ckpt/HUST/wx_inner_finetune'

res_dict = {}

for name in new_test[:]:

    stride = 1
    test_fea, test_lbl = [], []
    tmp_fea, tmp_lbl = all_loader[name]['fea'],all_loader[name]['lbl']
    tmp_fea_reversed = tmp_fea[::-1]
    test_fea.append(tmp_fea_reversed[::stride][::-1])
    tmp_lbl_reversed = tmp_lbl[::-1]
    test_lbl.append(tmp_lbl_reversed[::stride][::-1])
    test_fea = np.vstack(test_fea)
    test_lbl = np.vstack(test_lbl).squeeze()

    batch_size = 20 if len(test_fea)%20!=1 else 21
    rul_true, rul_pred, rul_base, SOH_TRUE, SOH_PRED, SOH_BASE = [], [], [], [], [], []
    for i in range(test_fea.shape[0] // batch_size + 1):

        test_fea_ = test_fea[i*batch_size: i*batch_size+batch_size].transpose(0,3,2,1)
        test_lbl_ = test_lbl[i*batch_size: i*batch_size+batch_size]
        testset = TensorDataset(torch.Tensor(test_fea_), torch.Tensor(test_lbl_))
        test_loader = DataLoader(testset, batch_size=batch_size,)
        if test_fea_.shape[0] == 0: continue

        model = CRNN(100,4,64,64)
        model = model.to(device)
        model.load_state_dict(torch.load(pretrain_model_path))

        _, y_pred, _, _, soh_pred = trainer.test(test_loader, model)
        rul_base.append(y_pred.cpu().detach().numpy())
        SOH_BASE.append(soh_pred.cpu().detach().numpy())
        print(f'test_time:{time.time()-tic}')

        for p in model.soh.parameters():
            p.requires_grad = False
        for p in model.rul.parameters():
            p.requires_grad = False
        for p in model.cnn.parameters():
            p.requires_grad = False

        tic = time.time()
        seed_torch(2021)

        num_epochs = 120
        trainer = FineTrainer(lr = 1e-4, n_epochs = num_epochs,device = device, patience = 1000,
                      lamda = lamda, train_alpha = train_alpha, valid_alpha = valid_alpha, model_name=finetune_model_path)
        model ,train_loss, valid_loss, total_loss, added_loss = trainer.train(test_loader, test_loader, model)
        print(f'finetuning_time:{time.time()-tic}')

        y_true, y_pred, mse_loss, soh_true, soh_pred = trainer.test(test_loader, model)
        rul_true.append(y_true.cpu().detach().numpy().reshape(-1,1))
        rul_pred.append(y_pred.cpu().detach().numpy())
        SOH_TRUE.append(soh_true.cpu().detach().numpy())
        SOH_PRED.append(soh_pred.cpu().detach().numpy())

    rul_true = np.vstack(rul_true).squeeze()
    rul_pred = np.vstack(rul_pred).squeeze()
    rul_base = np.vstack(rul_base).squeeze()
    SOH_TRUE = np.vstack(SOH_TRUE)
    SOH_PRED = np.vstack(SOH_PRED)
    SOH_BASE = np.vstack(SOH_BASE)

    
    res_dict.update({name:{
        'rul':{
            'true':rul_true[:]*rul_factor,
            'base':rul_base[:]*rul_factor,
            'transfer':rul_pred[:]*rul_factor,
        },
        'soh':{
            'true':SOH_TRUE[:,9]*cap_factor,
            'base':SOH_BASE[:,9]*cap_factor,
            'transfer':SOH_PRED[:,9]*cap_factor,
        },
                          }})
    
save_obj(res_dict,'./HAIRLAB/result/res_dict')


print('no problem')