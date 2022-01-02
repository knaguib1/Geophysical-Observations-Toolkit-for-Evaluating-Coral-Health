import os
import time
from datetime import *
import numpy as np
import torch
from models import *
from imblearn.over_sampling import SMOTE 
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import numpy as np
import pandas as pd

"""
    File name: utils.py
    Author: Kareem Naguib
    Date created: 12/01/2021
    Contact: knaguib1@gmail.com, knaguib3@gatech.edu
"""


def subset_data(df, dist_thresh, tar_feats, tar_class):
    """Data preparation for modeling """
    
    # 1.0 subset data based on distance threshold
    data = df[df.ac_distances < dist_thresh]
    
    # 2.0 select feature columns
    feat_cols = [c for c in df.columns if c[:4] == 'feat']
    
    # 2.1 Calipso feature selection
    if  tar_feats == 'calip_feats':
        idx = [i for i in range(10)] + [i for i in range(278, 300)] 
        idx += [i for i in range(205, 210)] + [i for i in range(220, 225)]
        feat_cols = list(np.array(feat_cols)[idx])
        
    # 2.2 calipso-giovani features
    if tar_feats == 'calip_giov':
        feat_cols += ['par', 'chlor_a' , 'ino_part']
        
        # filter for date delta between calipso and giovani 
        data['time_delta'] = abs(pd.to_datetime(data.calip_date) - pd.to_datetime(data.giov_date)).dt.days
        data = data[data['time_delta'] < 14]
    
    # 3.0 Drop na values and Calipso errors
    # drop baa na values
    data.dropna(subset=[tar_class], inplace=True)

    # drop calipso values < -100
    data = data[(data[feat_cols] >= -100).any(1)]
    
    return data, feat_cols
    
def class_balance(data, feat_cols, tar_class, tar_labels, sampling_method):
    """Split dataset into training/test using sampling method from config"""
    
    # create X and y datasets 
    subset_class = data[tar_class].isin(tar_labels)
    
    #1.0 Oversampling using SMOTE 
    if sampling_method == 'smote':
        X = data.loc[subset_class, feat_cols].values
        y = data.loc[subset_class, tar_class].values
        sm = SMOTE(random_state=42)
        X_sm, y_sm = sm.fit_resample(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm)
        
    # 2.0 Random undersampling
    elif sampling_method == 'undersampling':
        
        # number of data points to randomly sample
        coral_size = data[data[tar_class] == 1].shape[0]
        train_coral_size = 4*coral_size//5
        
        other_size = coral_size//4
        train_other_size = 4 * other_size // 5
        
        # randomly undersample each class that's not coral
        rock_idx = list(np.random.choice(data[data['class'] == 'Rock'].index, size=other_size))
        rubble_idx = list(np.random.choice(data[data['class'] == 'Rubble'].index, size=other_size))
        sand_idx = list(np.random.choice(data[data['class'] == 'Sand'].index, size=other_size))
        seagrass_idx = list(np.random.choice(data[data['class'] == 'Seagrass'].index, size=other_size))
        coral_idx = list(np.random.choice(data[data[tar_class] == 1].index, size=coral_size))

        # split train/test dataset using index
        train_rock = rock_idx[:train_other_size]
        train_rubble = rubble_idx[:train_other_size]
        train_sand = sand_idx[:train_other_size]
        train_seagrass =  seagrass_idx[:train_other_size]
        train_coral = coral_idx[:train_coral_size]

        test_rock = rock_idx[train_other_size:]
        test_rubble = rubble_idx[train_other_size:]
        test_sand = sand_idx[train_other_size:]
        test_seagrass =  seagrass_idx[train_other_size:]
        test_coral = coral_idx[train_coral_size:]

        # combine indices for test and train datasets
        test_idx = test_coral + test_rock + test_rubble + test_sand + test_seagrass
        train_idx = train_coral + train_rock + train_rubble + train_sand + train_seagrass

        X_train = data.loc[train_idx, feat_cols].values
        y_train = data.loc[train_idx, tar_class].values
        X_test = data.loc[test_idx, feat_cols].values
        y_test = data.loc[test_idx, tar_class].values
        
    return X_train, y_train, X_test, y_test


def scale_data(X_train, X_test, scale_method):
    
    if scale_method == 'MinMax':
      scaler = MinMaxScaler().fit(X_train)
      X_train_transformed = scaler.transform(X_train)
      X_test_transformed = scaler.transform(X_test)
    elif scale_method == 'MaxAbs':
      scaler = MaxAbsScaler().fit(X_train)
      X_train_transformed = scaler.transform(X_train)
      X_test_transformed = scaler.transform(X_test)
    elif scale_method == 'standard':
      scaler = StandardScaler().fit(X_train)
      X_train_transformed = scaler.transform(X_train)
      X_test_transformed = scaler.transform(X_test)
    
    else:
      X_train_transformed = X_train
      X_test_transformed = X_test
      
    return X_train_transformed, X_test_transformed
    
def model_selection(tar_feats, tar_labels, dropout=0.5):
    """Return FeedForwardNet model based on config """
    # dictionary of model parameters based on target features
    model_params = {
        'calip_only':(300, 128, 64, 32, 16), 
        'calip_feats':(42, 36, 24, 12, 6),
        'calip_giov':(303, 128, 64, 32, 16)
    }
    
    # feed forward neural net model parameters
    in_features, h1, h2, h3, h4 = model_params[tar_feats]
    out_features = len(tar_labels)
    
    # instantiate model
    model = FeedForwardNet(in_features, h1, h2, h3, h4, out_features, dropout=dropout).cuda()
    
    return model

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_batch_accuracy(output, target):
    """Computes the accuracy for a batch"""
    with torch.no_grad():

        batch_size = target.size(0)
        _, pred = output.max(1)
        correct = pred.eq(target).sum()

        return correct * 100.0 / batch_size


def train(model, device, data_loader, criterion, optimizer, epoch, print_freq=10):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if isinstance(input, tuple):
            input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
        else:
            input = input.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(input)
        target = target.long()  # this is casting int to long
        loss = criterion(output, target)
        assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'

        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        losses.update(loss.item(), target.size(0))
        accuracy.update(compute_batch_accuracy(output, target).item(), target.size(0))

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                epoch, i, len(data_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, acc=accuracy))

    return losses.avg, accuracy.avg


def evaluate(model, device, data_loader, criterion, print_freq=10):
    batch_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()

    results = []

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(data_loader):

            if isinstance(input, tuple):
                input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
            else:
                input = input.to(device)
            target = target.to(device)

            output = model(input)
            target = target.long()  # this is casting int to long
            loss = criterion(output, target)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            losses.update(loss.item(), target.size(0))
            accuracy.update(compute_batch_accuracy(output, target).item(), target.size(0))

            y_true = target.detach().to('cpu').numpy().tolist()
            y_pred = output.detach().to('cpu').max(1)[1].numpy().tolist()
            results.extend(list(zip(y_true, y_pred)))

            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                    i, len(data_loader), batch_time=batch_time, loss=losses, acc=accuracy))

    return losses.avg, accuracy.avg, results


