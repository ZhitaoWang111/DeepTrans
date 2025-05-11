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
from transformer import BatteryLifeTransformer
    
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import warnings
warnings.filterwarnings('ignore')

n_cyc = 100
in_stride = 10
fea_num = 100

v_low = 3.36
v_upp = 3.60
q_low = 610
q_upp = 1190
rul_factor = 3000
cap_factor = 1190

new_test  = ['9-6','4-5','1-2', '10-7','1-1', '6-1','6-6', '9-4','10-4','8-5', '5-3','10-6',
            '2-5','6-2','3-1','8-8', '8-1','8-6','7-6','6-8','7-5','10-1']

loader_path  = 'D:/wzt/bishe_data/data/HUST_cycle_dataloader/all_loader.pkl'
with open(loader_path, 'rb') as file:
    all_loader = pickle.load(file)

ign_bat = ['b25','b33','a0','a1','a2','a3','a4','b22','b24','b26','b38','b44','c25','c43','c44']
train_fea = load_obj('D:/wzt/bishe_data/data/MIT_data/fea_train')
train_lbl = load_obj('D:/wzt/bishe_data/data/MIT_data/label_train')
ne_train_name = list(train_fea.keys())

valid_fea = load_obj('D:/wzt/bishe_data/data/MIT_data/fea_test')
valid_lbl = load_obj('D:/wzt/bishe_data/data/MIT_data/label_test')
ne_valid_name = list(valid_fea.keys())

test_fea = load_obj('D:/wzt/bishe_data/data/MIT_data/fea_sec')
test_lbl = load_obj('D:/wzt/bishe_data/data/MIT_data/label_sec')
ne_test_name = list(test_fea.keys())

for name in ne_train_name:
    tmp_fea, tmp_lbl = train_fea.get(name), train_lbl.get(name)
    all_loader.update({name:{'fea':tmp_fea,'lbl':tmp_lbl}})
    
for name in ne_valid_name:
    tmp_fea, tmp_lbl = valid_fea.get(name), valid_lbl.get(name)
    all_loader.update({name:{'fea':tmp_fea,'lbl':tmp_lbl}})
    
for name in ne_test_name:
    tmp_fea, tmp_lbl = test_fea.get(name), test_lbl.get(name)
    all_loader.update({name:{'fea':tmp_fea,'lbl':tmp_lbl}})
    
for nm in list(all_loader.keys()):
    del_rows = []
    tmp_fea = all_loader[nm]['fea']
    tmp_lbl = all_loader[nm]['lbl']
    for i in range(1,11):
        del_rows += list(np.where(np.abs(np.diff(tmp_lbl[:,i])) > 0.05)[0])
    tmp_fea = np.delete(tmp_fea, del_rows, axis=0)
    tmp_lbl = np.delete(tmp_lbl, del_rows, axis=0)
    all_loader.update({nm:{'fea':tmp_fea,'lbl':tmp_lbl}})


stride = 10
train_fea, train_lbl = [], []
for name in ne_train_name + ne_test_name:
    if name in ign_bat:continue
    tmp_fea, tmp_lbl = all_loader[name]['fea'],all_loader[name]['lbl']
    tmp_fea_reversed = tmp_fea[::-1]
    train_fea.append(tmp_fea_reversed[::stride][::-1])
    tmp_lbl_reversed = tmp_lbl[::-1]
    train_lbl.append(tmp_lbl_reversed[::stride][::-1])
train_fea = np.vstack(train_fea)
train_lbl = np.vstack(train_lbl).squeeze()

stride = 10
valid_fea, valid_lbl = [], []
for name in ne_valid_name:
    if name in ign_bat:continue
    tmp_fea, tmp_lbl = all_loader[name]['fea'],all_loader[name]['lbl']
    tmp_fea_reversed = tmp_fea[::-1]
    valid_fea.append(tmp_fea_reversed[::stride][::-1])
    tmp_lbl_reversed = tmp_lbl[::-1]
    valid_lbl.append(tmp_lbl_reversed[::stride][::-1])
valid_fea = np.vstack(valid_fea)
valid_lbl = np.vstack(valid_lbl).squeeze()

print(train_fea.shape, train_lbl.shape, valid_fea.shape, valid_lbl.shape)

seed_torch(0)
batch_size = 256

train_fea_ = train_fea[:].copy()
train_lbl_ = train_lbl[:].copy()

train_fea_ = train_fea_.transpose(0,3,2,1)
valid_fea_ = valid_fea.transpose(0,3,2,1)

trainset = TensorDataset(torch.Tensor(train_fea_), torch.Tensor(train_lbl_))
validset = TensorDataset(torch.Tensor(valid_fea_), torch.Tensor(valid_lbl))

train_loader = DataLoader(trainset, batch_size=batch_size,shuffle=True)
valid_loader = DataLoader(validset, batch_size=batch_size,)

train_loader_t = DataLoader(trainset, batch_size=batch_size,shuffle=False)

'''
lamda (float): The weight of RUL loss
alpha (List: [float]): The weights of Capacity loss
'''         
lamda = 1e-2
alpha = torch.Tensor([0.1] * 10 )

seed_torch(0)
device = 'cuda'
model =  BatteryLifeTransformer()
model = model.to(device)
board_dir = '../ckpt/MIT_trans/runs'

num_epochs = 200
trainer = Trainer(lr = 8e-4, n_epochs = num_epochs,device = device, patience = 1600,
                  lamda = lamda, alpha = alpha, model_name='../ckpt/MIT_trans/mit_pretrain', board_dir=board_dir)
model ,train_loss, valid_loss, total_loss = trainer.train(train_loader, valid_loader, model)


lamda = 0.0
train_weight9 = [0.1] * 9
valid_weight9 = [0. if (i!=0) else 0.1 for i in train_weight9]
train_alpha = torch.Tensor(train_weight9 + [0.] )
valid_alpha = torch.Tensor(valid_weight9 + [0.])

pretrain_model_path = '../ckpt/MIT_trans/mit_pretrain_best.pt'
finetune_model_path = '../ckpt/MIT_trans/mit_finetune'

res_dict = {}

for name in new_test[:]:

    stride = 1
    test_fea, test_lbl = [], []
    if name in ign_bat: print('wrong bat')
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

        model =  BatteryLifeTransformer()
        model = model.to(device)
        model.load_state_dict(torch.load(pretrain_model_path))

        _, y_pred, _, _, soh_pred = trainer.test(test_loader, model)
        rul_base.append(y_pred.cpu().detach().numpy())
        SOH_BASE.append(soh_pred.cpu().detach().numpy())

        for p in model.feature_extractor.parameters():
            p.requires_grad = False

        seed_torch(2021)

        num_epochs = 100
        trainer = FineTrainer(lr = 1e-4, n_epochs = num_epochs,device = device, patience = 1000,
                      lamda = lamda, train_alpha = train_alpha, valid_alpha = valid_alpha, model_name=finetune_model_path, board_dir=board_dir)
        model ,train_loss, valid_loss, total_loss, added_loss = trainer.train(test_loader, test_loader, model)

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
                            }
                    })
    print(f'完成：{name}')
save_obj(res_dict,'../result/mit_trans_result')