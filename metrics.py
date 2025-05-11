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
    
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import warnings
warnings.filterwarnings('ignore')


result = load_obj('../result/HUST_result')

new_test  = ['9-6','4-5','1-2', '10-7','1-1', '6-1','6-6', '9-4','10-4','8-5', '5-3','10-6',
            '2-5','6-2','3-1','8-8', '8-1','8-6','7-6','6-8','7-5','10-1']

test_1 = ['1-1','7-5','9-4','10-1','10-4','10-7']
channel_1 = ['#1', '#50', '#65', '#70', '#73', '#76']

test_2 = ['1-2', '5-3','6-1','6-6','6-8']
channel_2 = ['#2','#34', '#39', '#44', '#45']

test_3 = ['2-5','4-5','7-6','8-1','8-8','9-6']
channel_3 = ['#12', '#28', '#51', '#54', '#61', '#67']

test_4 = ['3-1','6-2','8-5','8-6','10-6',]
channel_4 = ['#16', '#40', '#58', '#59', '#75']

fig = plt.figure(figsize=(12,12))
x_lim = 2000
plt.plot(range(x_lim), range(x_lim),'-',c='k', linewidth=3,label='y=x')
cmap = plt.get_cmap('YlGnBu')
norm = plt.Normalize(vmin=0.3, vmax=1)

colors = ['#09C988','b','#FDB137','PURPLE','#40CEE3','red',]

label_names = channel_1[:]
for i, name in enumerate(test_1):
    interval = 20

    rul_true = result[name]['rul']['true'] 
    rul_base = result[name]['rul']['base']
    rul_pred = result[name]['rul']['transfer']
    
    color = [len(rul_true)/2700]*len(rul_true)
    color = color[::interval]

    plt.plot(rul_true[::interval], rul_pred[::interval], '.',markersize=11,label=label_names[i], c=colors[i])
plt.legend(fontsize=25,loc="lower right")
x_major_locator=MultipleLocator(400)
y_major_locator=MultipleLocator(400)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
plt.xlim((0,x_lim))
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.tick_params(length=10)
plt.xticks([0,1000,2000],size=25)
plt.yticks([0,1000,2000],size=25)

plt.ylabel('Predicted RUL (cycles)',fontsize=30)
plt.xlabel('Actual RUL (cycles)',fontsize=30)
plt.show()
# plt.savefig(f'Figure/fig3/rul_1.png',dpi=1200,bbox_inches='tight')