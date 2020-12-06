#!/usr/bin/env python
# coding: utf-8

# Tabnet

#============================================================================================================

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
from sklearn.decomposition import PCA, TruncatedSVD, KernelPCA
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_tabnet.metrics import Metric
from pytorch_tabnet.tab_model import TabNetRegressor

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('../input/iterative-stratification/iterative-stratification-master')
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

#============================================================================================================

train_features = pd.read_csv('../input/lish-moa/train_features.csv')
train_targets_scored = pd.read_csv('../input/lish-moa/train_targets_scored.csv')
train_targets_nonscored = pd.read_csv('../input/lish-moa/train_targets_nonscored.csv')

test_features = pd.read_csv('../input/lish-moa/test_features.csv')
sample_submission = pd.read_csv('../input/lish-moa/sample_submission.csv')

#============================================================================================================

seed = 42

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASSED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
seed_everything(seed)

#============================================================================================================

# ## preprocess data

def drop_ctl_vehicle(train_features, test_features):
    
    train_features = train_features[train_features['cp_type'] != 'ctl_vehicle']
    test_features = test_features[test_features['cp_type'] != 'ctl_vehicle']
    target = train_targets_scored.iloc[train_features.index]
    train_features.reset_index(drop=True, inplace=True)
    test_features.reset_index(drop=True, inplace=True)
    
    return train_features, test_features, target

#============================================================================================================

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

#============================================================================================================

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

#============================================================================================================

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

#============================================================================================================

def var_thresh(threshold, train_features, test_features):

    data = pd.concat([train_features, test_features], ignore_index=True)
    cols_numeric = [f for f in data.columns if f not in ['sig_id', 'cp_type', 'cp_time', 'cp_dose']]
    mask = (data[cols_numeric].var() >= threshold).values
    tmp = data[cols_numeric].loc[:, mask]
    data = pd.concat([data[['sig_id', 'cp_type', 'cp_time', 'cp_dose']], tmp], axis=1)
    train_features = data.iloc[:train_features.shape[0], :].reset_index(drop=True)
    test_features = data.iloc[train_features.shape[0]:, :].reset_index(drop=True)
    
    return train_features, test_features

#============================================================================================================

def var_thresh_2(threshold, train_features, test_features):

    cols_numeric = [f for f in train_features.columns if f not in ['sig_id', 'cp_type', 'cp_time', 'cp_dose']]
    mask = (train_features[cols_numeric].var() >= 0.8).values
    train_features_ = train_features[cols_numeric].loc[:, mask]
    test_features_ = test_features[cols_numeric].loc[:, mask]
    train_features = pd.concat([train_features[['sig_id', 'cp_type', 'cp_time', 'cp_dose']], train_features_], axis=1).reset_index(drop=True)
    test_features = pd.concat([test_features[['sig_id', 'cp_type', 'cp_time', 'cp_dose']], test_features_], axis=1).reset_index(drop=True)

    return train_features, test_features

#============================================================================================================

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

#============================================================================================================

def scaling(train_features, test_features):
    
    scaler = RobustScaler()
    train_features_ = train_features.drop(['sig_id', 'cp_type', 'cp_time', 'cp_dose'], axis=1)
    columns_ = train_features_.columns
    train_features_numerical = scaler.fit_transform(train_features_)
    test_features_numerical = scaler.transform(test_features.drop(['sig_id', 'cp_type', 'cp_time', 'cp_dose'], axis=1))
    train_features_ = pd.DataFrame(train_features_numerical)
    test_features_ = pd.DataFrame(test_features_numerical)
    train_features_ = pd.concat([train_features[['sig_id', 'cp_type', 'cp_time', 'cp_dose']], train_features_], axis=1)
    test_features_ = pd.concat([test_features[['sig_id', 'cp_type', 'cp_time', 'cp_dose']], test_features_], axis=1)
    
    return train_features_, test_features_

#============================================================================================================

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

#============================================================================================================

def make_folds_old(train, n_splits):
    
    folds = train.copy()

    mskf = MultilabelStratifiedKFold(n_splits)

    for f, (t_idx, v_idx) in enumerate(mskf.split(X=train, y=target)):
        folds.loc[v_idx, 'kfold'] = int(f)

    folds['kfold'] = folds['kfold'].astype(int)
    
    return folds

#============================================================================================================

def preprocessor(train_features, test_features):
    
    print('start')
    
#     # drop_ctl_vehicle
#     train_features, test_features, target = drop_ctl_vehicle(train_features, test_features)
#     print('drop_ctl_vehicle: done')
    
    # rank_gauss
    train_features, test_features = rank_gauss(train_features, test_features)
    print('rank_gauss: done')
    print('train_features.shape: ', train_features.shape)
    print('test_features.shape: ', test_features.shape)
    
    # stats
    train_features, test_features = get_stats(train_features, test_features)
    print('get_stats: done')
    print('train_features.shape: ', train_features.shape)
    print('test_features.shape: ', test_features.shape)
    
    # pca
    train_features, test_features = get_pca(train_features, test_features, n_gs=600, n_cs=50) 
    print('get_pca: done')
    print('train_features.shape: ', train_features.shape)
    print('test_features.shape: ', test_features.shape)
    
    # var_thresh
    train_features, test_features = var_thresh_2(threshold=0.8, train_features=train_features, test_features=test_features)
    print('var_thresh: done')
    print('train_features.shape: ', train_features.shape)
    print('test_features.shape: ', test_features.shape)
    
#     # clustering
#     train_features, test_features = get_cluster(train_features, test_features, n_clusters_g=35, n_clusters_c=5, SEED=0)
#     print('get_clustering: done')
#     print('train_features.shape: ', train_features.shape)
#     print('test_features.shape: ', test_features.shape)
    
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

#============================================================================================================

folds = make_folds_old(train, n_splits=5)

#============================================================================================================

target_cols = target.drop('sig_id', axis=1).columns.values.tolist()
feature_cols = [c for c in pd.get_dummies(train, columns=['cp_time', 'cp_dose']).columns if c not in target_cols]
feature_cols = [c for c in feature_cols if c not in ['kfold', 'sig_id']]
len(feature_cols)

#============================================================================================================

# ### shape check

print(f'train.shape: {train.shape}')
print(f'folds.shape: {folds.shape}')
print(f'test.shape: {test.shape}')
print(f'target.shape: {target.shape}')
print(f'sample_submission.shape: {sample_submission.shape}')

#============================================================================================================

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

#============================================================================================================

# ### HyperParameters

DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 35
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-6
NFOLDS = 5
EARLY_STOPPING_STEPS = 10
EARLY_STOP = False
smoothing = 1e-5
p_min = smoothing
p_max = 1 - smoothing

num_features = len(feature_cols)
num_targets = len(target_cols)
hidden_size_1 = 1024
hidden_size_2 = 1024

#============================================================================================================

MAX_EPOCH = 200
# n_d and n_a are different from the original work, 32 instead of 24
# This is the first change in the code from the original
tabnet_params = dict(
    n_d = 32,
    n_a = 32,
    n_steps = 1,
    gamma = 1.3,
    lambda_sparse = 0,
    optimizer_fn = optim.Adam,
    optimizer_params = dict(lr = 2e-2, weight_decay = 1e-5),
    mask_type = "entmax",
    scheduler_params = dict(
        mode = "min", patience = 5, min_lr = 1e-5, factor = 0.9),
    scheduler_fn = optim.lr_scheduler.ReduceLROnPlateau,
    seed = seed,
    verbose = 10
)

#============================================================================================================

# ### training function

def train_fn(model, optimizer, scheduler, loss_tr, loss_fn, dataloader, device):
    model.train()
    final_loss = 0
    final_metric = 0
    
    for data in dataloader:
        optimizer.zero_grad()
        inputs, targets = data['x'].to(device), data['y'].to(device)
        outputs = model(inputs)
        loss = loss_tr(outputs, targets)
        metric = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        final_loss += loss.item()
        final_metric += metric.item()
        
    final_loss /= len(dataloader)
    final_metric /= len(dataloader)
    
    return final_loss, final_metric

#============================================================================================================

def valid_fn(model, loss_tr, loss_fn, dataloader, device):
    model.eval()
    final_loss = 0
    final_metric = 0
    valid_preds = []
    
    for data in dataloader:
        
        inputs, targets = data['x'].to(device), data['y'].to(device)
        outputs = model(inputs)
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

#============================================================================================================

def inference_fn(model, dataloader, device):
    model.eval()
    preds = []
    
    for data in dataloader:
        inputs = data['x'].to(device)
        
        with torch.no_grad():
            outputs = model(inputs)
            
        outputs = torch.clamp(torch.sigmoid(outputs).detach().cpu(), p_min, p_max)    
        preds.append(outputs.numpy())
        
    preds = np.concatenate(preds)
        
    return preds

#============================================================================================================

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

#============================================================================================================

class LogitsLogLoss(Metric):
    """
    LogLoss with sigmoid applied
    """

    def __init__(self):
        self._name = "logits_ll"
        self._maximize = False

    def __call__(self, y_true, y_pred):
        """
        Compute LogLoss of predictions.

        Parameters
        ----------
        y_true: np.ndarray
            Target matrix or vector
        y_score: np.ndarray
            Score matrix or vector

        Returns
        -------
            float
            LogLoss of predictions vs targets.
        """
        
        logits = 1 / (1 + np.exp(-y_pred))
        logits = np.clip(logits, p_min, p_max)
        aux = (1 - y_true) * np.log(1 - logits + 1e-15) + y_true * np.log(logits + 1e-15)
        return np.mean(-aux)

#============================================================================================================

# def run_training(fold, seed, EPOCHS, LEARNING_RATE):
    
#     print(f'==========FOLD{fold+1}==========')
    
    
#     seed_everything(seed)
    
#     train_losses = list()
#     valid_losses = list()
    
#     train = pd.get_dummies(folds, columns=['cp_time', 'cp_dose'])
#     test_ = pd.get_dummies(test, columns=['cp_time', 'cp_dose'])
    
#     trn_idx = train[train['kfold'] != fold].index
#     val_idx = train[train['kfold'] == fold].index
    
#     train_df = train[train['kfold'] != fold].reset_index()
#     valid_df = train[train['kfold'] == fold].reset_index()
    
#     x_train, y_train = train_df[feature_cols].values, train_df[target_cols].values
#     x_valid, y_valid = valid_df[feature_cols].values, valid_df[target_cols].values
    
# #     train_dataset = MoADataset(x_train, y_train)
# #     valid_dataset = MoADataset(x_valid, y_valid)
    
# #     trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# #     validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
#     model = TabNetRegressor(**tabnet_params)
    
#     model.fit(
#         X_train = X_train,
#         y_train = y_train,
#         eval_set = [(x_valid, y_valid)],
#         eval_name = ["val"],
#         eval_metric = ["logits_ll"],
#         max_epochs = MAX_EPOCH,
#         patience = 20,
#         batch_size = 1024, 
#         virtual_batch_size = 32,
#         num_workers = 1,
#         drop_last = False,
#         # To use binary cross entropy because this is not a regression problem
#         loss_fn = SmoothCrossEntropyLoss(smoothing=smoothing)
#     )
    

# #     model.to(DEVICE)
    
#     torch.backends.cudnn.benchmark = True
    
# #     optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
# #     scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.3, div_factor=1e3, 
# #                                               max_lr=1e-2, epochs=EPOCHS, steps_per_epoch=len(trainloader)) 
    
# #     loss_fn = lambda inputs, targets : F.binary_cross_entropy((torch.clamp(torch.sigmoid(inputs), p_min, p_max)), targets)
# #     loss_tr = SmoothCrossEntropyLoss(smoothing=smoothing)
    
# #     early_stoppping_steps = EARLY_STOPPING_STEPS
# #     early_step = 0
    
#     oof = np.zeros((len(train), target.iloc[:, 1:].shape[1]))
# #     best_loss = np.inf

#     ### Predict on validation ###
#     preds_val = model.predict(X_val)
    
# #     for epoch in range(EPOCHS):
        
# #         print(f'-----EPOCH{epoch+1}-----')
        
# #         train_loss, train_metric = train_fn(model, optimizer, scheduler, loss_tr, loss_fn, trainloader, DEVICE)
# #         print(f'train_loss: {train_loss:.5f}, train_metric: {train_metric:.5f}')
# #         train_losses.append(train_loss)
# #         valid_loss, valid_metric, valid_preds = valid_fn(model, loss_tr, loss_fn, validloader, DEVICE)
# #         print(f'valid_loss: {valid_loss:.5f}, valid_metric: {valid_metric:.5f}')
# #         valid_losses.append(valid_loss)
        
# #         if valid_loss < best_loss:
            
# #             best_loss = valid_loss
# #             oof[val_idx] = valid_preds
# #             torch.save(model.state_dict(), f'Simple_FOLD{fold+1}_SEED{seed}.pth')
            
# #         elif(EARLY_STOP == True):
            
# #             early_step += 1
# #             if (early_step >= early_stopping_steps):
# #                 break
                
# #     plt.plot(train_losses, label='train_losses')
# #     plt.plot(valid_losses, label='valid_losses')
# #     plt.xlabel('epochs')
# #     plt.ylabel('loss')
# #     plt.ylim([1e-2, 2e-2])
# #     plt.title(f'fold{fold+1} losses')
# #     plt.show()
                
#     #-----------------------PREDICTION-------------------------
                
#     x_test = test_[feature_cols].values
#     testdataset = TestDataset(x_test)
#     testloader = torch.utils.data.DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=False)
    
#     model = Model_Simple(num_features=num_features, 
#                              num_targets=num_targets, 
#                              hidden_size_1=hidden_size_1,
#                              hidden_size_2=hidden_size_2)
    
#     model.load_state_dict(torch.load(f'Simple_FOLD{fold+1}_SEED{seed}.pth'))
#     model.to(DEVICE)
    
#     predictions = np.zeros((len(test_), target.iloc[:, 1:].shape[1]))
#     predictions = inference_fn(model, testloader, DEVICE)
    
#     return oof, predictions

#============================================================================================================

# ### Single fold training

scores_auc_all = []
test_cv_preds = []

# mskf = MultilabelStratifiedKFold(n_splits = NB_SPLITS, random_state = 0, shuffle = True)

oof_preds = []
oof_targets = []
scores = []
scores_auc = []

oof = np.zeros((len(train), target.iloc[:, 1:].shape[1]))

SEED = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

for seed in SEED:
    
    print(f'=====SEED: {seed}=====')
    
    oof_tmp = np.zeros((len(train), target.iloc[:, 1:].shape[1]))
    
    for fold in range(NFOLDS):
        print("FOLDS: ", fold + 1)
        print('*' * 60)

        train_ = pd.get_dummies(folds, columns=['cp_time', 'cp_dose'])
        test_ = pd.get_dummies(test, columns=['cp_time', 'cp_dose'])

        trn_idx = train_[train_['kfold'] != fold].index
        val_idx = train_[train_['kfold'] == fold].index

        train_df = train_[train_['kfold'] != fold].reset_index()
        valid_df = train_[train_['kfold'] == fold].reset_index()
    
        x_train, y_train = train_df[feature_cols].values, train_df[target_cols].values
        x_valid, y_valid = valid_df[feature_cols].values, valid_df[target_cols].values

        x_test = test_[feature_cols].values

        ### Model ###
        model = TabNetRegressor(**tabnet_params)

        ### Fit ###
        # Another change to the original code
        # virtual_batch_size of 32 instead of 128
#         model.fit(
#             X_train = x_train,
#             y_train = y_train,
#             eval_set = [(x_valid, y_valid)],
#             eval_name = ["val"],
#             eval_metric = ["logits_ll"],
#             max_epochs = MAX_EPOCH,
#             patience = 20,
#             batch_size = 1024, 
#             virtual_batch_size = 32,
#             num_workers = 1,
#             drop_last = False,
#             # To use binary cross entropy because this is not a regression problem
#             loss_fn = SmoothCrossEntropyLoss(smoothing=smoothing)
#         )

#         !cp -r ../input/tabnet_models/{str(seed)}_{str(fold)}/* .
#         !zip {seed}_{fold}.zip model_params.json network.pt
        
        model.load_model(f'./TabNet_FOLD{fold+1}_SEED{seed}.zip')
        
        print('-' * 60)

        ### Predict on validation ###
        preds_val = model.predict(x_valid)
        # Apply sigmoid to the predictions
        preds = 1 / (1 + np.exp(-preds_val))
        preds = np.clip(preds, p_min, p_max)
        oof_tmp[val_idx] += preds
#         score = np.min(model.history["val_logits_ll"])

        ### Save OOF for CV ###
        oof_preds.append(preds)
        oof_targets.append(y_valid)
#         scores.append(score)

        ### Predict on test ###
        preds_test = model.predict(x_test)
        preds_test = 1 / (1 + np.exp(-preds_test))
        preds_test = np.clip(preds_test, p_min, p_max)
#         predictions_tmp += preds_test / NFOLDS
        test_cv_preds.append(preds_test)

#         name = f'TabNet_FOLD{fold+1}_SEED{seed}'
#         model.save_model(name)
    
    oof += oof_tmp / len(SEED)

train[target_cols] = oof
test_preds_all = np.stack(test_cv_preds)

#============================================================================================================

y_true = train_targets_scored[target_cols].values

train[target_cols] = oof
valid_results = train_targets_scored.drop(columns=target_cols).merge(
    train[['sig_id']+target_cols], on='sig_id', how='left').fillna(0)
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

#============================================================================================================

file_path = 'oof_TabNet.npy'
np.save(file_path, y_pred)

all_feat = [col for col in sample_submission.columns if col not in ["sig_id"]]
# To obtain the same length of test_preds_all and submission
data_path = "../input/lish-moa/"
test = pd.read_csv(data_path + "test_features.csv")
sig_id = test[test["cp_type"] != "ctl_vehicle"].sig_id.reset_index(drop = True)
tmp = pd.DataFrame(test_preds_all.mean(axis = 0), columns = all_feat)
tmp["sig_id"] = sig_id

#============================================================================================================

submission = pd.merge(test[["sig_id"]], tmp, on = "sig_id", how = "left")
submission.fillna(0, inplace = True)

#============================================================================================================

submission.to_csv("submission_TabNet.csv", index = None)
submission.head()



