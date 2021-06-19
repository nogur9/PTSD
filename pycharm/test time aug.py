#!/usr/bin/env python
# coding: utf-8

# In[39]:


from load_data import load_data
# compare amount of noise added to examples created during the test-time augmentation
from numpy.random import seed
from numpy.random import normal
from numpy import arange
from numpy import mean
from numpy import std
from scipy.stats import mode
from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import precision_score, roc_auc_score, f1_score, recall_score, precision_recall_curve, auc, \
    average_precision_score, precision_recall_fscore_support
from xgboost import XGBClassifier
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[40]:


def print_results(y_test, y_pred):
    # print(f1_score(y_test.astype(int), y_pred))
    precision, recall, _ = precision_recall_curve(y_test.astype(int), y_pred)
    avs = average_precision_score(y_test.astype(int), y_pred)
    print(f"average_precision_score = {avs}")

    auc_score = auc(recall, precision)
    print(f"pr_auc = {auc_score}")
    # plt.plot(recall, precision)
    #plt.show()
    print(f"holdout i = , roc_auc = {roc_auc_score(y_test.astype(int), y_pred)}")



# In[41]:


# create a test set for a row of real data with an unknown label
def create_test_set(row, n_cases=3, feature_scale=0.2):
    test_set = list()
    test_set.append(row)
    # make copies of row
    for _ in range(n_cases):
        # create vector of random gaussians
        gauss = normal(loc=0.0, scale=feature_scale, size=len(row))
        # add to test case
        new_row = row + gauss
        # store in test set
        test_set.append(new_row)
    return test_set


# make predictions using test-time augmentation
def test_time_augmentation(model, X_test, noise):
    # evaluate model
    y_hat = list()
    for i in range(X_test.shape[0]):
        # retrieve the row
        row = X_test.iloc[i]
        # create the test set
        test_set = create_test_set(row, feature_scale=noise)

        # make a prediction for all examples in the test set
        labels = model.predict_proba(np.array(test_set))[:, 1]
        # select the label as the mode of the distribution
        label, _ = mode(labels)
        # store the prediction
        y_hat.append(label)
    return y_hat


# In[42]:


df_preprocessed, features, target_feature = load_data()
df_preprocessed = df_preprocessed.dropna(subset=['target_binary_intrusion'], how='any')

X, X_out, Y, y_out = train_test_split(df_preprocessed[features], df_preprocessed['target_binary_intrusion'],
                                      test_size=0.15, stratify=df_preprocessed['target_binary_intrusion'])

cv = StratifiedKFold(5)

# In[44]:


# evaluate different number of synthetic examples created at test time
noise = arange(0.01, 0.31, 0.1)
results = list()
for n in noise:
    print('\n\n\n',n)
    # initialize numpy random number generator
    seed(1)
    # create dataset

    for train, test in cv.split(X, Y):
        x_train, y_train = X.iloc[train], Y.iloc[train]
        x_test, y_test = X.iloc[test], Y.iloc[test]

        # create the model
        model = XGBClassifier(class_weights=[1, 5])
        model.fit(x_train.values, y_train)

        y_pred = test_time_augmentation(model, x_train, n)
        print_results(y_train, y_pred)

        y_pred = test_time_augmentation(model, x_test, n)
        print_results(y_test, y_pred)

# In[ ]:
