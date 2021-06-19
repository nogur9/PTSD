#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
from keras.layers import Dense
from tensorflow import keras
#from tensorflow.keras import layers
#import tensorflow_datasets as tfds


from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import precision_score, roc_auc_score, f1_score, recall_score, precision_recall_curve, auc, average_precision_score, precision_recall_fscore_support
from xgboost import XGBClassifier
import pandas as pd
from load_data import load_data
import matplotlib.pyplot as plt


import tensorflow as tf
from tensorflow.keras.models import Sequential

from keras_fsl.models.head_models import DenseSigmoid
from keras_fsl.layers import GramMatrix, Classification
from keras_fsl.losses.gram_matrix_losses import BinaryCrossentropy
from keras_fsl.metrics.gram_matrix_metrics import classification_accuracy, min_eigenvalue
from keras_fsl.utils.tensors import get_dummies
from tensorflow.keras.layers import Input, Dense


from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam



# In[2]:


learning_rate = 0.003
meta_step_size = 0.25

inner_batch_size = 25
eval_batch_size = 25

meta_iters = 2000
eval_iters = 5
inner_iters = 4

eval_interval = 1
train_shots = 20
shots = 5
classes = 5


# In[42]:

# In[15]:
def print_results(y_true, y_pred, name=''):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    avs = average_precision_score(y_true, y_pred)
    auc_score = auc(recall, precision)

    print('/n/n', name)

    plt.plot(recall, precision)
    plt.title("recall precision curve")

    plt.show()

    plt.hist(y_pred)
    plt.title("prediction histogram")
    plt.show()

    print(f"pr_auc = {auc_score}")

    print(f"average_precision_score = {avs}")

    print(f"holdout i = , roc_auc = {roc_auc_score(y_true, y_pred)}")


# In[16]:



    
df_preprocessed, features, target_feature = load_data()
df_preprocessed = df_preprocessed.dropna(subset = ['target_binary_intrusion'], how='any')
    
    
    
X, X_out, Y, y_out = train_test_split(df_preprocessed[features], df_preprocessed['target_binary_intrusion'],                                          test_size=0.15,                                          stratify=df_preprocessed['target_binary_intrusion'])


cv = StratifiedKFold(5)

for train, test in cv.split(X, Y):
    x_train, y_train = X.iloc[train], Y.iloc[train]
    x_test, y_test = X.iloc[test], Y.iloc[test]
    x_support, x_query, y_support, y_query = train_test_split(x_train, y_train, test_size=0.15,
                                                              stratify=y_train)

    # create the model
    input_shape = x_support.shape[1]  # first shape is batch_size

    # %% Training
    encoder = Dense(73)
    support_layer = GramMatrix("DenseSigmoid")
    model = Sequential([encoder, support_layer])
    model.compile(optimizer="Adam", loss=BinaryCrossentropy(), metrics=[classification_accuracy(), min_eigenvalue])
    model.fit(x=x_support, y=y_support, validation_data=([x_query], [y_query]), epochs=5)

    # model = XGBClassifier(class_weights=[1,5])
    # model.fit(x_train, y_train)ddd

    y_pred = model.predict_proba(x_test)[:, 1]
    print_results(y_test, y_pred)

    # y_pred = test_time_augmentation(model, x_test, n)
    # print_results(y_test, y_pred)
    
    #model = XGBClassifier(class_weights=[1,5])
    #model.fit(x_train, y_train)


    y_pred = model.predict_proba(x_test)[:,1]
    print_results(y_test, y_pred)
        
   # y_pred = test_time_augmentation(model, x_test, n)
    #print_results(y_test, y_pred)


# In[34]:


encoder.input_shape


# In[36]:


support_layer


# In[ ]:




