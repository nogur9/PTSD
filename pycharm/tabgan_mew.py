#!/usr/bin/env python
# coding: utf-8

# In[1]:



import os
import glob

import numpy as np
import tensorflow as tf

from preprocessing import PCL_calculator, cv_preprocessing

import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras import layers
#import tensorflow_datasets as tfds


from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import precision_score, roc_auc_score, f1_score, recall_score, precision_recall_curve, auc, average_precision_score, precision_recall_fscore_support
#from xgboost import XGBClassifier
import pandas as pd
from load_data import load_data
import matplotlib.pyplot as plt
from tabgan import sampler
#OriginalGenerator, GANGenerator
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import precision_score, roc_auc_score, auc, average_precision_score, precision_recall_curve


# In[2]:


def print_results(y_true, y_pred, name = ''):
    
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    avs = average_precision_score(y_true, y_pred)
    auc_score = auc(recall, precision)
    
    print ('\n\n', name)

    plt.plot(recall, precision)
    plt.title("recall precision curve")

    plt.show()
    
    plt.hist(y_pred)
    plt.title("prediction histogram")
    plt.show()

    print(f"pr_auc = {auc_score}")

    print(f"average_precision_score = {avs}")

    print(f"holdout i = , roc_auc = {roc_auc_score(y_true, y_pred)}")
        


# In[3]:



    
df_preprocessed, features, target_feature = load_data()
df_preprocessed = df_preprocessed.dropna(subset = ['target_binary_intrusion'], how='any')
    

X, X_out, Y, y_out = train_test_split(df_preprocessed[features], df_preprocessed['target_binary_intrusion'],
                                      test_size=0.15,
                                      stratify=df_preprocessed['target_binary_intrusion'])

X = X.drop(["ID"], axis=1)

cv = StratifiedKFold(3)


# In[4]:



for train, test in cv.split(X, Y):
    x_train, y_train = X.iloc[train], Y.iloc[train]
    x_test, y_test = X.iloc[test], Y.iloc[test]

    x_train, x_test = cv_preprocessing(x_train, x_test)

    x_train['stupid'] = y_train.values
    x_test['stupid'] = y_test.values
    y_train = pd.DataFrame(x_train['stupid'])
    y_test = pd.DataFrame(x_test['stupid'])

    x_train = pd.DataFrame(x_train.drop('stupid', axis=1))
    x_test = pd.DataFrame(x_test.drop('stupid', axis=1))

    #generate data
    new_train1, new_target1 = sampler.OriginalGenerator().generate_data_pipe(x_train, y_train, x_test)#, only_adversarial=True)
    new_train2, new_target2 = sampler.GANGenerator().generate_data_pipe(x_train, y_train, x_test)#, only_adversarial=True)

    # example with all params defined
    new_train3, new_target3 = sampler.GANGenerator(gen_x_times=1.1, cat_cols=None, bot_filter_quantile=0.001,
                                       top_filter_quantile=0.999,
                                       is_post_process=True,
                                       adversaial_model_params={
                                           "metrics": "AUC", "max_depth": 2,
                                           "max_bin": 100, "n_estimators": 500,
                                           "learning_rate": 0.02, "random_state": 42,
                                       }, pregeneration_frac=2,
                                       epochs=500).generate_data_pipe(x_train, y_train,
                                                                      x_test, deep_copy=True,
                                                                      only_adversarial=False,
                                                                      use_adversarial=True)
    # create the model

    #%% Training

    model = CatBoostClassifier(class_weights={0:1,1:5})
    model.fit(new_train3,new_target3)
    y_pred = model.predict_proba(x_test)[:,1]
    print_results(y_test, y_pred)
        
   # y_pred = test_time_augmentation(model, x_test, n)
    #print_results(y_test, y_pred)

