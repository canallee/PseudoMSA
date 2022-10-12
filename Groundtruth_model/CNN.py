# pytorch version of the CNN model, with outputing regression instead 
# of binary classification. Model from the paper 
# 'Deep diversification of an AAV capsid protein by machine learning'
# https://github.com/google-research/google-research/tree/master/aav 

import torch
import torch.nn as nn
from utils.torch_utils import *
import torch.nn.functional as F
import numpy as np

# negative log likelihood loss
def NLL_loss(y_true, y_pred):
    mean = y_pred[:, 0]
    variance = F.softplus(y_pred[:, 1]) + 1e-6
    log_variance = torch.log(variance)
    #print(mean, variance, log_variance)
    part1 = 0.5 * log_variance.mean(dim=-1)
    part2 = 0.5 * (torch.square(y_true-mean)/variance).mean(dim=-1)
    part3 = 0.5 * np.log(2*np.pi)
    NLL_loss = part1+part2+part3
    return NLL_loss


class CNN_ground(nn.Module):
    def __init__(self, hidden_conv=12, hidden_fc=64, pool_width=2,
                 seq_len=512, n_chars=20):
        super(CNN_ground, self).__init__()
        # pre flatten:
        self.conv1 = nn.Conv1d(n_chars, hidden_conv,
                               pool_width, stride=1, padding='same')
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        self.maxpool1 = nn.MaxPool1d(pool_width, stride=pool_width)
        self.conv2 = nn.Conv1d(hidden_conv, hidden_conv//2, 
                               pool_width, stride=1, padding='same')
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)
        self.maxpool2 = nn.MaxPool1d(pool_width, stride=pool_width)
        self.conv_block = nn.Sequential(
            self.conv1,
            #nn.BatchNorm1d(hidden_conv),
            nn.ReLU(),
            self.maxpool1,
            self.conv2,
            #nn.BatchNorm1d(hidden_conv//2),
            nn.ReLU(),
            self.maxpool2
        )
        
        # post flatten:
        self.fc1 = nn.Linear((seq_len//2)*(hidden_conv//4), hidden_fc)
        self.fc2 = nn.Linear(hidden_fc, hidden_fc//2)
        self.fc3 = nn.Linear(hidden_fc//2, 2)
        self.fc_block = nn.Sequential(
            self.fc1,
            #nn.BatchNorm1d(hidden_fc),
            nn.ReLU(),
            self.fc2,
            #nn.BatchNorm1d(hidden_fc//2),
            nn.ReLU(),
            self.fc3
        )
        
    def forward(self, x):
        x = x.transpose(1,2)
        x = self.conv_block(x)
        #x = x.transpose(1,2)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.fc_block(x)
        return x

class CNN_ground_AAV(nn.Module):
    def __init__(self, hidden_conv=12, hidden_fc=64, pool_width=2,
                 seq_len=512, n_chars=21):
        super(CNN_ground_AAV, self).__init__()
        # pre flatten:
        self.conv1 = nn.Conv1d(n_chars, hidden_conv,
                               pool_width, stride=1, padding='same')
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        self.maxpool1 = nn.MaxPool1d(pool_width, stride=pool_width)
        self.conv2 = nn.Conv1d(hidden_conv, hidden_conv//2, 
                               pool_width, stride=1, padding='same')
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)
        self.maxpool2 = nn.MaxPool1d(pool_width, stride=pool_width)
        self.conv3 = nn.Conv1d(hidden_conv//2, hidden_conv//4, 
                               pool_width, stride=1, padding='same')
        nn.init.xavier_uniform_(self.conv3.weight.data, 1.)
        self.maxpool3 = nn.MaxPool1d(pool_width, stride=pool_width)
        
        self.conv_block = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.maxpool1,
            self.conv2,
            nn.ReLU(),
            self.maxpool2,
            self.conv3,
            nn.ReLU(),
            self.maxpool3,
        )
        
        # post flatten:
        self.fc1 = nn.Linear(((seq_len//4)*(hidden_conv//4))//2, hidden_fc)
        self.fc2 = nn.Linear(hidden_fc, hidden_fc//2)
        self.fc3 = nn.Linear(hidden_fc//2, 2)
        self.fc_block = nn.Sequential(
            self.fc1,
            #nn.BatchNorm1d(hidden_fc),
            nn.ReLU(),
            self.fc2,
            #nn.BatchNorm1d(hidden_fc//2),
            nn.ReLU(),
            self.fc3
        )
        
    def forward(self, x):
        x = x.transpose(1,2)
        #print(x.shape)
        x = self.conv_block(x)
        #print(x.shape)
        #x = x.transpose(1,2)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.fc_block(x)
        return x