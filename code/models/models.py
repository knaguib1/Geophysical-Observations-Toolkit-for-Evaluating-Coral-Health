import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import roc_curve, auc
from imblearn.over_sampling import SMOTE 
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

import torch.nn as nn
import torch.nn.functional as F

"""
    File name: models.py
    Author: Kareem Naguib
    Date created: 12/01/2021
    Contact: knaguib1@gmail.com, knaguib3@gatech.edu
"""

class FeedForwardNet(nn.Module):
    def __init__(self, in_features, h1, h2, h3, h4, out_features, dropout = 0.5):
        super(FeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc4 = nn.Linear(h3, h4)
        self.output_layer = nn.Linear(h4, out_features)
        
        # hyper parameters
        self.drop_out = nn.Dropout(p = dropout)
        self.batch_norm1 = nn.BatchNorm1d(in_features)
        self.batch_norm2 = nn.BatchNorm1d(h1)
        self.batch_norm3 = nn.BatchNorm1d(h2)
        self.batch_norm4 = nn.BatchNorm1d(h3)

    def forward(self, x):
        x = F.relu(self.drop_out(self.fc1(self.batch_norm1(x))))
        x = F.relu(self.drop_out(self.fc2(self.batch_norm2(x))))
        x = F.relu(self.drop_out(self.fc3(self.batch_norm3(x))))
        x = F.relu(self.drop_out(self.fc4(self.batch_norm4(x))))
        x = self.output_layer(x)
    
        return x

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(6, 16, 5)
        self.fc1 = nn.Linear(in_features=16*72, out_features=128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        self.drop_out = nn.Dropout(p = 0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.drop_out(self.conv1(x))))
        x = self.pool(F.relu(self.drop_out(self.conv2(x))))
        x = x.view(-1, 16*72)
        x = F.relu(self.drop_out(self.fc1(x)))
        x = F.relu(self.drop_out(self.fc2(x)))
        x = self.fc3(x)
        return x