#!/usr/bin/env python
# coding: utf-8

# In[14]:


from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import precision_score, roc_auc_score, f1_score, recall_score, precision_recall_curve, auc, average_precision_score, precision_recall_fscore_support
from load_data import load_data
from sklearn.preprocessing import StandardScaler

import random
import numpy as np
import torch
import learn2learn as l2l

from torch import nn, optim

from torchmeta.toy import Sinusoid
from torchmeta.utils.data import BatchMetaDataLoader
import matplotlib.pyplot as plt
from fancyimpute import IterativeImputer
import numpy as np
# In[26]:


class SineModel(nn.Module):

    def __init__(self, dim):
        super().__init__()
        #self.hidden1 = nn.Linear(dim, 1)
        self.hidden2 = nn.Linear(dim, 3)
        self.hidden3 = nn.Linear(3, 1)
        
    def forward(self, x):
        #x = nn.functional.relu(self.hidden1(x))
        x = nn.functional.relu(self.hidden2(x))
        x = self.hidden3(x)
        return x


# In[40]:



shots=10
tasks_per_batch=16
num_tasks=160000
adapt_lr=0.01
meta_lr=0.001
adapt_steps=10
hidden_dim=73

    
df_preprocessed, features, target_feature = load_data()
df_preprocessed = df_preprocessed.dropna(subset = ['target_binary_intrusion'], how='any')
    
    
    
X, X_out, Y, y_out = train_test_split(df_preprocessed[features], df_preprocessed['target_binary_intrusion'],                                          test_size=0.15,                                          stratify=df_preprocessed['target_binary_intrusion'])


cv = StratifiedKFold(5)


for train, test in cv.split(X, Y):
    x_train, y_train = X.iloc[train], Y.iloc[train]
    x_test, y_test = X.iloc[test], Y.iloc[test]

    # create the model
    model = SineModel(dim=x_train.shape[1])
    maml = l2l.algorithms.MAML(model, lr=float(adapt_lr), allow_unused=True)
    opt = optim.Adam(maml.parameters(), float(meta_lr))
    lossfn = nn.MSELoss()

    meta_train_loss = 0.0

    # # for each task in the batch
    # opt.zero_grad()
    # meta_train_loss.backward()
    # opt.step()
    learner = maml.clone()
            
    x_support,  x_query, y_support, y_query = train_test_split(x_train, y_train, test_size=0.25,
                                          stratify=y_train)
            
    ss = StandardScaler()
    x_support = ss.fit_transform(x_support)
    x_query = ss.transform(x_query)
    x_test = ss.transform(x_test)

    mice = IterativeImputer(max_iter=1000)
    x_support = mice.fit_transform(x_support)
    x_query = mice.fit_transform(x_query)
    x_test = mice.fit_transform(x_test)
    for _ in range(adapt_steps): # adaptation_steps
        support_preds = learner(torch.from_numpy(x_support).float())
        support_loss = lossfn(support_preds.float(), torch.from_numpy(y_support.values.reshape(-1, 1))).float()
        learner.adapt(support_loss)

    query_preds = learner(x_query)
    query_loss = lossfn(query_preds, y_query)
    meta_train_loss += query_loss


    opt.zero_grad()
    meta_train_loss.backward()
    opt.step()

    y_pred = model.predict_proba(x_train)[:,1]# * model3.predict_proba(x_test)[:,1]

    precision, recall, _ = precision_recall_curve(y_train.astype(int), y_pred)
    avs = average_precision_score(y_train.astype(int), y_pred)
    print(f"average_precision_score = {avs}")

    auc_score = auc(recall, precision)
    print(f"pr_auc = {auc_score}")
    plt.plot(recall, precision)
    plt.show()
    print(f"holdout i = , roc_auc = {roc_auc_score(y_train.astype(int), y_pred)}")
        
    y_pred = model.predict_proba(x_test)[:,1]# * model3.predict_proba(x_test)[:,1]

    #print(f1_score(y_test.astype(int), y_pred))
    precision, recall, _ = precision_recall_curve(y_test.astype(int), y_pred)
    avs = average_precision_score(y_test.astype(int), y_pred)
    print(f"average_precision_score = {avs}")

    auc_score = auc(recall, precision)
    print(f"pr_auc = {auc_score}")
    plt.plot(recall, precision)
    plt.show()
    print(f"holdout i = , roc_auc = {roc_auc_score(y_test.astype(int), y_pred)}")
    print('f')
            

