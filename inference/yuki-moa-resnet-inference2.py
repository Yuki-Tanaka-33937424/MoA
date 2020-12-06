#!/usr/bin/env python
# coding: utf-8

import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import os
import copy
import seaborn as sns
sns.set()

from sklearn import preprocessing
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('../input/iterative-stratification/iterative-stratification-master')
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

#======================================================================================================

train_features = pd.read_csv('../input/lish-moa/train_features.csv')
train_targets_scored = pd.read_csv('../input/lish-moa/train_targets_scored.csv')
train_targets_nonscored = pd.read_csv('../input/lish-moa/train_targets_nonscored.csv')
train_drug = pd.read_csv('../input/lish-moa/train_drug.csv')

test_features = pd.read_csv('../input/lish-moa/test_features.csv')
sample_submission = pd.read_csv('../input/lish-moa/sample_submission.csv')

#======================================================================================================

GENES = [col for col in train_features.columns if col.startswith('g-')]
CELLS = [col for col in train_features.columns if col.startswith('c-')]
predictors = GENES+CELLS

#======================================================================================================

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASSED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
seed_everything(seed=42)

#======================================================================================================

# ## preprocess data

def drop_ctl_vehicle(train_features, test_features):
    
    train_features = train_features[train_features['cp_type'] != 'ctl_vehicle']
    test_features = test_features[test_features['cp_type'] != 'ctl_vehicle']
    target = train_targets_scored.iloc[train_features.index]
    train_features.reset_index(drop=True, inplace=True)
    test_features.reset_index(drop=True, inplace=True)
    
    return train_features, test_features, target

#======================================================================================================

def rank_gauss(train_features, test_features):
    
    train_features_ = train_features.copy()
    test_features_ = test_features.copy()
    
    GENES = [col for col in train_features_.columns if col.startswith('g-')]
    CELLS = [col for col in train_features_.columns if col.startswith('c-')]
    
    for col in (GENES + CELLS):

        transformer = QuantileTransformer(n_quantiles=100,random_state=0, output_distribution="normal")
        vec_len = len(train_features_[col].values)
        vec_len_test = len(test_features_[col].values)
        raw_vec = train_features_[col].values.reshape(vec_len, 1)
        transformer.fit(raw_vec)

        train_features_[col] = transformer.transform(raw_vec).reshape(1, vec_len)[0]
        test_features_[col] = transformer.transform(test_features_[col].values.reshape(vec_len_test, 1)).reshape(1, vec_len_test)[0]
        
    return train_features_, test_features_

#======================================================================================================

def get_stats(train_features, test_features):
    
    GENES = [col for col in train_features.columns if col.startswith('g-')]
    CELLS = [col for col in train_features.columns if col.startswith('c-')]
    
    for df in [train_features, test_features]:
#         df['g_sum'] = df[GENES].sum(axis = 1)
        df['g_mean'] = df[GENES].mean(axis = 1)
        df['g_std'] = df[GENES].std(axis = 1)
        df['g_kurt'] = df[GENES].kurtosis(axis = 1)
        df['g_skew'] = df[GENES].skew(axis = 1)
#         df['g_max'] = df[GENES].max(axis=1)
#         df['g_min'] = df[GENES].max(axis=1)
#         df['c_sum'] = df[CELLS].sum(axis=1)
        df['c_mean'] = df[CELLS].mean(axis = 1)
        df['c_std'] = df[CELLS].std(axis = 1)
        df['c_kurt'] = df[CELLS].kurtosis(axis = 1)
        df['c_skew'] = df[CELLS].skew(axis = 1)
#         df['c_max'] = df[CELLS].max(axis=1)
#         df['c_min'] = df[CELLS].min(axis=1)
#         df['gc_sum'] = df[GENES + CELLS].sum(axis = 1)
        df['gc_mean'] = df[GENES + CELLS].mean(axis = 1)
        df['gc_std'] = df[GENES + CELLS].std(axis = 1)
        df['gc_kurt'] = df[GENES + CELLS].kurtosis(axis = 1)
        df['gc_skew'] = df[GENES + CELLS].skew(axis = 1)
        
    return train_features, test_features

#======================================================================================================

def get_pca(train_features, test_features, n_gs, n_cs):
    
    GENES = [col for col in train_features.columns if col.startswith('g-')]
    CELLS = [col for col in train_features.columns if col.startswith('c-')]
    
    pca_gs = PCA(n_components = n_gs)
    pca_cs = PCA(n_components = n_cs)
    
    train_pca_gs = pca_gs.fit_transform(train_features[GENES])
    train_pca_cs = pca_cs.fit_transform(train_features[CELLS])
    test_pca_gs = pca_gs.transform(test_features[GENES])
    test_pca_cs = pca_cs.transform(test_features[CELLS])
    
    train_pca_gs = pd.DataFrame(train_pca_gs, columns=[f'pca_G-{i}' for i in range(n_gs)])
    train_pca_cs = pd.DataFrame(train_pca_cs, columns=[f'pca_C-{i}' for i in range(n_cs)])
    test_pca_gs = pd.DataFrame(test_pca_gs, columns=[f'pca_G-{i}' for i in range(n_gs)])
    test_pca_cs = pd.DataFrame(test_pca_cs, columns=[f'pca_C-{i}' for i in range(n_cs)])
    
    train_features = pd.concat([train_features, train_pca_gs, train_pca_cs], axis=1)
    test_features = pd.concat([test_features, test_pca_gs, test_pca_cs], axis=1)
    
    return train_features, test_features

#======================================================================================================

def var_thresh(threshold, train_features, test_features):

    cols_numeric = [f for f in train_features.columns if f not in ['sig_id', 'cp_type', 'cp_time', 'cp_dose']]
    mask = (train_features[cols_numeric].var() >= 0.8).values
    train_features_ = train_features[cols_numeric].loc[:, mask]
    test_features_ = test_features[cols_numeric].loc[:, mask]
    train_features = pd.concat([train_features[['sig_id', 'cp_type', 'cp_time', 'cp_dose']], train_features_], axis=1).reset_index(drop=True)
    test_features = pd.concat([test_features[['sig_id', 'cp_type', 'cp_time', 'cp_dose']], test_features_], axis=1).reset_index(drop=True)

    return train_features, test_features

#======================================================================================================

def var_thresh(threshold, train_features, test_features):

    cols_numeric = [f for f in train_features.columns if f not in ['sig_id', 'cp_type', 'cp_time', 'cp_dose']]
    mask = (train_features[cols_numeric].var() >= 0.8).values
    train_features_ = train_features[cols_numeric].loc[:, mask]
    test_features_ = test_features[cols_numeric].loc[:, mask]
    train_features = pd.concat([train_features[['sig_id', 'cp_type', 'cp_time', 'cp_dose']], train_features_], axis=1).reset_index(drop=True)
    test_features = pd.concat([test_features[['sig_id', 'cp_type', 'cp_time', 'cp_dose']], test_features_], axis=1).reset_index(drop=True)

    return train_features, test_features

#======================================================================================================

def get_cluster(train, test, n_clusters_g = 35, n_clusters_c = 5, SEED = 0):
    
    GENES = [col for col in train.columns if col.startswith('g-')]
    CELLS = [col for col in train.columns if col.startswith('c-')]
    
    def create_cluster(train, test, features, kind = 'g', n_clusters = n_clusters_g):
        
        train_ = train[features].copy()
        test_ = test[features].copy()
        data = pd.concat([train_, test_], axis = 0)
        kmeans = KMeans(n_clusters = n_clusters, random_state = SEED).fit(data)
        train[f'clusters_{kind}'] = kmeans.labels_[:train.shape[0]]
        test[f'clusters_{kind}'] = kmeans.labels_[train.shape[0]:]
        train = pd.get_dummies(train, columns = [f'clusters_{kind}'])
        test = pd.get_dummies(test, columns = [f'clusters_{kind}'])
        
        return train, test
    
    train, test = create_cluster(train, test, GENES, kind = 'g', n_clusters = n_clusters_g)
    train, test = create_cluster(train, test, CELLS, kind = 'c', n_clusters = n_clusters_c)
    
    return train, test

#======================================================================================================

def scaling(train_features, test_features):
    
    scaler = RobustScaler()
    train_features_ = train_features.drop(['sig_id', 'cp_type', 'cp_time', 'cp_dose'], axis=1)
    columns_ = train_features_.columns
    train_features_numerical = scaler.fit_transform(train_features_)
    test_features_numerical = scaler.transform(test_features.drop(['sig_id', 'cp_type', 'cp_time', 'cp_dose'], axis=1))
    train_features_ = pd.DataFrame(train_features_numerical, columns=columns_)
    test_features_ = pd.DataFrame(test_features_numerical, columns=columns_)
    train_features = pd.concat([train_features[['sig_id', 'cp_type', 'cp_time', 'cp_dose']], train_features_], axis=1)
    test_features = pd.concat([test_features[['sig_id', 'cp_type', 'cp_time', 'cp_dose']], test_features_], axis=1)
    
    return train_features, test_features

#======================================================================================================

def make_folds(train, num_starts, num_splits):
    
    train_ = train.copy()
    folds = []

    # LOAD FILES
    train_feats = pd.read_csv('../input/lish-moa/train_features.csv')
    scored = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')
    drug = pd.read_csv('/kaggle/input/lish-moa/train_drug.csv')
    scored = scored.loc[train_feats['cp_type'] == 'trt_cp', :]
    drug = drug.loc[train_feats['cp_type'] == 'trt_cp', :]
    targets = scored.columns[1:]
    scored = scored.merge(drug, on='sig_id', how='left') 

    # LOCATE DRUGS
    vc = scored.drug_id.value_counts()
    vc1 = vc.loc[(vc <= 6) | (vc == 12) | (vc == 18)].index.sort_values()
    vc2 = vc.loc[(vc > 6) & (vc != 12) & (vc != 18)].index.sort_values()

    for seed in range(num_starts):

        # STRATIFY DRUGS 18X OR LESS
        dct1 = {}; dct2 = {}
        skf = MultilabelStratifiedKFold(n_splits = num_splits, shuffle = True, random_state = seed)
        tmp = scored.groupby('drug_id')[targets].mean().loc[vc1]
        for fold,(idxT,idxV) in enumerate(skf.split(tmp,tmp[targets])):
            dd = {k:fold for k in tmp.index[idxV].values}
            dct1.update(dd)

        # STRATIFY DRUGS MORE THAN 18X
        skf = MultilabelStratifiedKFold(n_splits = num_splits, shuffle = True, random_state = seed)
        tmp = scored.loc[scored.drug_id.isin(vc2)].reset_index(drop = True)
        for fold,(idxT,idxV) in enumerate(skf.split(tmp,tmp[targets])):
            dd = {k:fold for k in tmp.sig_id[idxV].values}
            dct2.update(dd)

        # ASSIGN FOLDS
        scored['fold'] = scored.drug_id.map(dct1)
        scored.loc[scored.fold.isna(),'fold'] =            scored.loc[scored.fold.isna(),'sig_id'].map(dct2)
        scored.fold = scored.fold.astype('int8')
        folds.append(scored.fold.values)

        del scored['fold']
        
        for i in range(len(folds)):
            train_[f'seed{i}'] = folds[i]

    return train_

#======================================================================================================

def make_folds_old(train, n_splits):
    
    folds = train.copy()

    mskf = MultilabelStratifiedKFold(n_splits)

    for f, (t_idx, v_idx) in enumerate(mskf.split(X=train, y=target)):
        folds.loc[v_idx, 'kfold'] = int(f)

    folds['kfold'] = folds['kfold'].astype(int)
    
    return folds

#======================================================================================================

def preprocessor(train_features, test_features):
    
#     # drop_ctl_vehicle
#     train_features, test_features, target = drop_ctl_vehicle(train_features, test_features)
    
    # rank_gauss
    train_features, test_features = rank_gauss(train_features, test_features)
    print('rank_gauss: done')
    print('train_features.shape', train_features.shape)
    print('test_features.shape', test_features.shape)
    
    # stats
    train_features, test_features = get_stats(train_features, test_features)
    print('get_stats: done')
    print('train_features.shape', train_features.shape)
    print('test_features.shape', test_features.shape)
    
    # pca
    train_features, test_features = get_pca(train_features, test_features, n_gs=600, n_cs=50) 
    print('get_pca: done')
    print('train_features.shape', train_features.shape)
    print('test_features.shape', test_features.shape)
    
    # var_thresh
    train_features, test_features = var_thresh(threshold=0.8, train_features=train_features, test_features=test_features)
    print('var_thresh: done')
    print('train_features.shape', train_features.shape)
    print('test_features.shape', test_features.shape)
    
#     # clustering
#     train_features, test_features = get_cluster(train_features, test_features, n_clusters_g=35, n_clusters_c=5, SEED=0)
#     print('clustering: done')
#     print('train_features.shape', train_features.shape)
#     print('test_features.shape', test_features.shape)
    
#     # Scaling
#     train_features, test_features = scaling(train_features, test_features)
#     print('scaling: done')
    
    # data merge
    train = train_features.merge(train_targets_scored, on='sig_id')
    train = train[train['cp_type'] != 'ctl_vehicle'].reset_index(drop=True)
    test = test_features[test_features['cp_type'] != 'ctl_vehicle'].reset_index(drop=True)

    target = train[train_targets_scored.columns]
    
    train = train.drop('cp_type', axis=1)
    test = test.drop('cp_type', axis=1)
    
    return train, target, test  

train, target, test = preprocessor(train_features, test_features)

#======================================================================================================

folds = make_folds_old(train, n_splits=5)

#======================================================================================================

target_cols = target.drop('sig_id', axis=1).columns.values.tolist()
feature_cols = [c for c in pd.get_dummies(train, columns=['cp_time', 'cp_dose']).columns if c not in target_cols]
feature_cols = [c for c in feature_cols if c not in ['kfold', 'sig_id']]
len(feature_cols)

#======================================================================================================

# ### shape check
print(f'train.shape: {train.shape}')
print(f'folds.shape: {folds.shape}')
print(f'test.shape: {test.shape}')
print(f'target.shape: {target.shape}')
print(f'sample_submission.shape: {sample_submission.shape}')

#======================================================================================================

# ## Dataset Classes

class MoADataset:
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
        
    def __len__(self):
        return (self.features.shape[0])
    
    def __getitem__(self, idx):
        dct = {
            'x' : torch.tensor(self.features[idx, :], dtype=torch.float), 
            'y' : torch.tensor(self.targets[idx, :], dtype=torch.float)
        }
        return dct
    
class TestDataset:
    def __init__(self, features):
        self.features = features
        
    def __len__(self):
        return (self.features.shape[0])
    
    def __getitem__(self, idx):
        dct = {
            'x' : torch.tensor(self.features[idx, :], dtype=torch.float)
        }
        return dct

#======================================================================================================

# ### training function

def train_fn(model, optimizer, scheduler, loss_tr, loss_fn, dataloader, device):
    rejected = 0
    model.train()
    final_loss = 0
    final_metric = 0
    
    for data in dataloader:
        optimizer.zero_grad()
        inputs, targets = data['x'].to(device), data['y'].to(device)
        if len(inputs) > 1: 
            inputs1 = inputs[:, :-len(predictors)]
            inputs2 = inputs[:, -len(predictors):]
            outputs = model(inputs1, inputs2)
            loss = loss_tr(outputs, targets)
            metric = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()
        
            final_loss += loss.item()
            final_metric += metric.item()
            
        else:
            rejected += len(inputs)
        
    final_loss /= (len(dataloader) - rejected)
    final_metric /= (len(dataloader) - rejected)
    
    return final_loss, final_metric

def valid_fn(model, loss_tr, loss_fn, dataloader, device):
    model.eval()
    final_loss = 0
    final_metric = 0
    valid_preds = []
    
    for data in dataloader:
        
        inputs, targets = data['x'].to(device), data['y'].to(device)
        inputs1 = inputs[:, :-len(predictors)]
        inputs2 = inputs[:, -len(predictors):]
        outputs = model(inputs1, inputs2)
        loss = loss_tr(outputs, targets)
        metric = loss_fn(outputs, targets)
        final_loss += loss.item()
        final_metric += metric.item()
        outputs = torch.clamp(torch.sigmoid(outputs).detach().cpu(), p_min, p_max)
        valid_preds.append(outputs.numpy())
        
    final_loss /= len(dataloader)
    final_metric /= len(dataloader)
    valid_preds = np.concatenate(valid_preds)
        
    return final_loss, final_metric, valid_preds

def inference_fn(model, dataloader, device):
    model.eval()
    preds = []
    
    for data in dataloader:
        inputs = data['x'].to(device)
        inputs1 = inputs[:, :-len(predictors)]
        inputs2 = inputs[:, -len(predictors):]
        
        with torch.no_grad():
            outputs = model(inputs1, inputs2)
            
        outputs = torch.clamp(torch.sigmoid(outputs).detach().cpu(), p_min, p_max)    
        preds.append(outputs.numpy())
        
    preds = np.concatenate(preds)
        
    return preds

#======================================================================================================

# ### loss_function

import torch
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F

class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth(targets, n_classes, smoothing=0.0):
        assert 0 <= smoothing <= 1
        with torch.no_grad():
            targets = targets * (1 - smoothing) + torch.ones_like(targets).to(DEVICE) * smoothing / n_classes
        return targets

    def forward(self, inputs, targets):
        targets = SmoothCrossEntropyLoss()._smooth(targets, inputs.shape[1], self.smoothing)

        if self.weight is not None:
            inputs = inputs * self.weight.unsqueeze(0)

        loss = F.binary_cross_entropy_with_logits(inputs, targets)

        return loss

#======================================================================================================

# ### HyperParameters

DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 35
BATCH_SIZE = 128
BATCH_SIZE_nonscored = 512
LEARNING_RATE = 1e-2
WEIGHT_DECAY = 1e-5
NFOLDS = 5
EARLY_STOPPING_STEPS = 10
EARLY_STOP = False
EARLY_STOPPING_STEPS_nonscored = 4
EARLY_STOP_nonscored = True
smoothing = 0.001
p_min = smoothing
p_max = 1 - smoothing

num_features_1 = len(feature_cols)
num_features_2 = len(predictors)
num_targets = len(target_cols)
hidden_1 = 256
hidden_2 = 256

#======================================================================================================

# ### Model

class Model_ResNet(nn.Module):
    def __init__(self, num_features_1, num_features_2, num_targets, hidden_1, hidden_2):
        super(Model_ResNet, self).__init__()
        
        self.batchnorm1_1 = nn.BatchNorm1d(num_features_1)
        self.dropout1_1 = nn.Dropout(0.3)
#         self.Linear1_1 = nn.utils.weight_norm(nn.Linear(num_features_1, hidden_1))
        self.Linear1_1 = nn.Linear(num_features_1, hidden_1)
        self.relu1_1 = nn.ReLU()
        self.batchnorm1_2 = nn.BatchNorm1d(hidden_1)
#         self.dropout1_2 = nn.Dropout(0.3)
#         self.Linear1_2 = nn.utils.weight_norm(nn.Linear(hidden_1, hidden_2))
        self.Linear1_2 = nn.Linear(hidden_1, hidden_2)
        self.relu1_2 = nn.ReLU()
        
        self.batchnorm2_1 = nn.BatchNorm1d(num_features_2+hidden_2)
        self.dropout2_1 = nn.Dropout(0.35)
#         self.Linear2_1 = nn.utils.weight_norm(nn.Linear(num_features_2+hidden_2, hidden_1))
        self.Linear2_1 = nn.Linear(num_features_2+hidden_2, hidden_1)
        self.relu2_1 = nn.ReLU()
        self.batchnorm2_2 = nn.BatchNorm1d(hidden_1)
#         self.dropout2_2 = nn.Dropout(0.3)
#         self.Linear2_2 = nn.utils.weight_norm(nn.Linear(hidden_1, hidden_1))
        self.Linear2_2 = nn.Linear(hidden_1, hidden_1)
        self.relu2_2 = nn.ReLU()
        self.batchnorm2_3 = nn.BatchNorm1d(hidden_1)
#         self.dropout2_3 = nn.Dropout(0.3)
#         self.Linear2_3 = nn.utils.weight_norm(nn.Linear(hidden_1, hidden_2))
        self.Linear2_3 = nn.Linear(hidden_1, hidden_2)
        self.relu2_3 = nn.ReLU()
        
        self.batchnorm3_1 = nn.BatchNorm1d(hidden_2)
#         self.dropout3_1 = nn.Dropout(0.3)
#         self.Linear3_1 = nn.utils.weight_norm(nn.Linear(hidden_2, hidden_2))
        self.Linear3_1 = nn.Linear(hidden_2, hidden_2)
        self.relu3_1 = nn.ReLU()
        self.batchnorm3_2 = nn.BatchNorm1d(hidden_2) 
#         self.dropout3_2 = nn.Dropout(0.3)
#         self.Linear3_2 = nn.utils.weight_norm(nn.Linear(hidden_2, num_targets))
        self.Linear3_2 = nn.Linear(hidden_2, num_targets)
        
        
        
    def recalibrate_layer(self, layer):

        if(torch.isnan(layer.weight_v).sum() > 0):
            print ('recalibrate layer.weight_v')
            layer.weight_v = torch.nn.Parameter(torch.where(torch.isnan(layer.weight_v), torch.zeros_like(layer.weight_v), layer.weight_v))
            layer.weight_v = torch.nn.Parameter(layer.weight_v + 1e-7)

        if(torch.isnan(layer.weight).sum() > 0):
            print ('recalibrate layer.weight')
            layer.weight = torch.where(torch.isnan(layer.weight), torch.zeros_like(layer.weight), layer.weight)
            layer.weight += 1e-7
            
    def forward(self, x_1, x_2):
        
        input_1 = x_1
        input_2 = x_2
        
        input_3 = self.batchnorm1_1(input_1)
        input_3 = self.dropout1_1(input_3)
#         self.recalibrate_layer(layer=self.Linear1_1)
        input_3 = self.Linear1_1(input_3)
        input_3 = self.relu1_1(input_3)
        input_3 = self.batchnorm1_2(input_3)
#         input_3 = self.dropout1_2(input_3)
#         self.recalibrate_layer(layer=self.Linear1_2)
        input_3 = self.Linear1_2(input_3)
        input_3 = self.relu1_2(input_3)
        
        input_3_concat = torch.cat([input_2, input_3], axis=1)
        
        input_4 = self.batchnorm2_1(input_3_concat)
        input_4 = self.dropout2_1(input_4)
#         self.recalibrate_layer(layer=self.Linear2_1)
        input_4 = self.Linear2_1(input_4)
        input_4 = self.relu2_1(input_4)
        input_4 = self.batchnorm2_2(input_4)
#         input_4 = self.dropout2_2(input_4)
#         self.recalibrate_layer(layer=self.Linear2_2)
        input_4 = self.Linear2_2(input_4)
        input_4 = self.relu2_2(input_4)
        input_4 = self.batchnorm2_3(input_4)
#         input_4 = self.dropout2_3(input_4)
#         self.recalibrate_layer(layer=self.Linear2_3)
        input_4 = self.Linear2_3(input_4)
        input_4 = self.relu2_3(input_4)
        
        input_4_avg = (input_3 + input_4) * 0.5
        
        output = self.batchnorm3_1(input_4_avg)
#         output = self.dropout3_1(output)
#         self.recalibrate_layer(layer=self.Linear3_1)
        output = self.Linear3_1(output)
        output = self.relu3_1(output)
        output = self.batchnorm3_2(output)
#         output = self.dropout3_2(output)
#         self.recalibrate_layer(layer=self.Linear3_2)
        output = self.Linear3_2(output)
        
        return output
#======================================================================================================

# ### Single fold training

def run_training(fold, seed, EPOCHS, LEARNING_RATE):
    
    print(f'==========FOLD{fold+1}==========')
    
    
    seed_everything(seed)
    
    train_losses = list()
    valid_losses = list()
    
    train = pd.get_dummies(folds, columns=['cp_time', 'cp_dose'])
    test_ = pd.get_dummies(test, columns=['cp_time', 'cp_dose'])
    
    trn_idx = train[train['kfold'] != fold].index
    val_idx = train[train['kfold'] == fold].index
    
    train_df = train[train['kfold'] != fold].reset_index()
    valid_df = train[train['kfold'] == fold].reset_index()
    
    x_train, y_train = train_df[feature_cols].values, train_df[target_cols].values
    x_valid, y_valid = valid_df[feature_cols].values, valid_df[target_cols].values
    
    x_train2 = train_df[predictors].values
    x_valid2 = valid_df[predictors].values
    
    x_train = np.concatenate([x_train, x_train2], axis=1)
    x_valid = np.concatenate([x_valid, x_valid2], axis=1)
    
    train_dataset = MoADataset(x_train, y_train)
    valid_dataset = MoADataset(x_valid, y_valid)
    
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    

    model = Model_ResNet(num_features_1=num_features_1,
                      num_features_2=num_features_2, 
                      num_targets=num_targets, 
                      hidden_1=hidden_1,
                      hidden_2=hidden_2)
    
    model.to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.3, div_factor=1e3, 
                                              max_lr=1e-2, epochs=EPOCHS, steps_per_epoch=len(trainloader)) 
    
    loss_fn = lambda inputs, targets : F.binary_cross_entropy((torch.clamp(torch.sigmoid(inputs), p_min, p_max)), targets)
    loss_tr = SmoothCrossEntropyLoss(smoothing=smoothing)
    
    early_stoppping_steps = EARLY_STOPPING_STEPS
    early_step = 0
    
    oof = np.zeros((len(train), target.iloc[:, 1:].shape[1]))
    best_loss = np.inf
    
#     for epoch in range(EPOCHS):
        
#         print(f'-----EPOCH{epoch+1}-----')
        
#         train_loss, train_metric = train_fn(model, optimizer, scheduler, loss_tr, loss_fn, trainloader, DEVICE)
#         print(f'train_loss: {train_loss:.5f}, train_metric: {train_metric:.5f}')
#         train_losses.append(train_loss)
#         valid_loss, valid_metric, valid_preds = valid_fn(model, loss_tr, loss_fn, validloader, DEVICE)
#         print(f'valid_loss: {valid_loss:.5f}, valid_metric: {valid_metric:.5f}')
#         valid_losses.append(valid_loss)
        
#         if valid_loss < best_loss:
            
#             best_loss = valid_loss
#             oof[val_idx] = valid_preds
#             torch.save(model.state_dict(), f'ResNet_FOLD{fold+1}_SEED{seed}.pth')
            
#         elif(EARLY_STOP == True):
            
#             early_step += 1
#             if (early_step >= early_stopping_steps):
#                 break
                
#     y_true = train_targets_scored[target_cols].values

#     cv = 0
#     for i in range(len(target_cols)):
#         cv_ = log_loss(y_true[val_idx, i], oof[val_idx, i], labels=[0, 1])
#         cv += cv_ / len(target_cols)
                
#     plt.plot(train_losses, label='train_losses')
#     plt.plot(valid_losses, label='valid_losses')
#     plt.xlabel('epochs')
#     plt.ylabel('loss')
#     plt.ylim([1e-2, 2e-2])
#     plt.title(f'fold{fold+1} losses')
#     plt.show()
                
    #-----------------------PREDICTION-------------------------
                
    x_test = test_[feature_cols].values
    x_test2 = test_[predictors].values
    
    x_test = np.concatenate([x_test, x_test2], axis=1)
    
    testdataset = TestDataset(x_test)
    testloader = torch.utils.data.DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = Model_ResNet(num_features_1=num_features_1,
                      num_features_2=num_features_2, 
                      num_targets=num_targets, 
                      hidden_1=hidden_1,
                      hidden_2=hidden_2)
    
    model.load_state_dict(torch.load(f'../input/yuki-moa-resnet1/ResNet_FOLD{fold+1}_SEED{seed}.pth'))
    model.to(DEVICE)
    
    valid_loss, valid_metric, valid_preds = valid_fn(model, loss_tr, loss_fn, validloader, DEVICE)
    oof[val_idx] = valid_preds
    predictions = np.zeros((len(test_), target.iloc[:, 1:].shape[1]))
    predictions = inference_fn(model, testloader, DEVICE)
    
    return oof, predictions

#======================================================================================================

def run_k_fold(NFOLD, seed, EPOCH, LEARNING_RATE):
    oof = np.zeros((len(train), len(target_cols)))
    predictions = np.zeros((len(test), len(target_cols)))
#     CVs = list()
    
    for fold in range(NFOLDS):
        oof_, pred_ = run_training(fold, seed, EPOCH, LEARNING_RATE)
        
        predictions += pred_ / NFOLDS
        oof += oof_
        
    return oof, predictions

#======================================================================================================

# Averaging on multiple SEEDS

SEED = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
oof = np.zeros((len(train), len(target_cols)))
predictions = np.zeros((len(test), len(target_cols)))
# CVs = list()

print(f'used device: {DEVICE}')

for seed in SEED:
    
    print(f' ')
    print(f'SEED : {seed}')
    print(f'')
    
    oof_, predictions_ = run_k_fold(NFOLDS, seed, EPOCHS, LEARNING_RATE)
    oof += oof_ / len(SEED)
    predictions += predictions_ / len(SEED)
    
train[target_cols] = oof
test[target_cols] = predictions

#======================================================================================================

valid_results = train_targets_scored.drop(columns=target_cols).merge(
    train[['sig_id']+target_cols], on='sig_id', how='left').fillna(0)

y_true = train_targets_scored[target_cols].values
y_pred = valid_results[target_cols].values

cv = 0
for i in range(len(target_cols)):
    cv_ = log_loss(y_true[:, i], y_pred[:, i])
    cv += cv_ / len(target_cols)
    
auc = 0
for i in range(len(target_cols)):
    auc_ = roc_auc_score(y_true[:, i], y_pred[:, i])
    auc += auc_ / len(target_cols)
    
print(f'CV log_loss: {cv:.6f}')
print(f'AUC: {auc:.6f}')

#======================================================================================================

file_path = 'oof_ResNet.npy'
np.save(file_path, y_pred)

sub = sample_submission.drop(columns=target_cols).merge(
    test[['sig_id']+target_cols], on='sig_id', how='left').fillna(0)
sub.to_csv('submission_ResNet.csv', index=False)

#======================================================================================================

print(f'sample_submission.shape : {sample_submission.shape}')
print(f'sub.shape : {sub.shape}')

