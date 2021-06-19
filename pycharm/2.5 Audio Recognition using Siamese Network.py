#!/usr/bin/env python
# coding: utf-8

# # Audio Recognition Using Siamese Network

# In the last tutorial, we saw how to use the siamese networks to recognize a face. Now we will see how to use the siamese networks to recognize the audio. We will train our network to differentiate between a dog's and a cat's sound. The dataset of cats and dogs audio can be downloaded from here https://www.kaggle.com/mmoreaux/audio-cats-and-dogs#cats_dogs.zip
# 
# Once we have downloaded the data, we fragment our data into three folders Dogs, Sub_dogs, and Cats. In Dogs and Sub_dogs, we place the dog's barking audio and in Cats folder, we place the cat's audio. The objective of our network is to recognize whether the audio is the dog's barking sound or some different sound. As we know for a Siamese network, we need to feed input as a pair, we select an audio from Dogs and Sub_dogs folder and mark it as a genuine pair and we select an audio from Dogs and Cats folder and mark it as an imposite pair. That is, (dogs, subdogs) is genuine pair and (dogs, cats) is imposite pair.
# 
# Now we will step by step how to train our siamese network to recognize whether the audio is the dog's barking sound or some different sound. First, We will load all the necessary libraries:

# In[1]:
from preprocessing import cv_preprocessing

print("sad imports")

#basic imports
import glob
#import IPython
from random import randint
print("sad imports")

#data processing
import librosa
import numpy as np
print("sad imports")
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import precision_score, roc_auc_score, auc, average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt

#modelling
from sklearn.model_selection import train_test_split
print("sad imports")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras import backend as K
print("sad imports")

from keras.layers import Activation
from keras.layers import Input, Lambda, Dense, Dropout, Flatten
from keras.models import Model
print("sad imports")

from keras.optimizers import RMSprop
from load_data import load_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Before going ahead, We load and listen to the audio clips,

print("yay imports")
#
# IPython.display.Audio("data/audio/Dogs/dog_barking_0.wav")
#
#
# # In[3]:
#
#
# IPython.display.Audio("data/audio/Cats/cat_13.wav")
#

# So, how can we feed this raw audio to our network? How can we extract meaningful features from the raw audio? As we know neural networks accept only vectorized input, we need to convert our audio to a feature vector. How can we do that? Well, there are several mechanisms through which we can generate embeddings for the audio. One such popular mechanism is Mel-Frequency Cepstral Coefficients (MFCC). 
# 
# We use MFCC for vectorizing our audio. MFCC converts the short-term power spectrum of an audio using a linear cosine transform of a log power spectrum on a nonlinear mel scale of frequency. To learn more about MFCC check this nice tutorial (http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/).
# 
# We will use MFCC function from librosa library for generating the audio embeddings. So, we define a function called audio2vector which return the audio embeddings given the audio:

# In[4]:


def audio2vector(file_path, max_pad_len=400):
    
    #read the audio file
    audio, sr = librosa.load(file_path, mono=True)
    #reduce the shape
    audio = audio[::3]
    
    #extract the audio embeddings using MFCC
    mfcc = librosa.feature.mfcc(audio, sr=sr) 
    
    #as the audio embeddings length varies for different audio, we keep the maximum length as 400
    #pad them with zeros
    pad_width = max_pad_len - mfcc.shape[1]
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfcc


# We will load one audio file and see the embeddings

# In[5]:


audio_file = 'data/audio/Dogs/dog_barking_0.wav'


# In[6]:


#audio2vector(audio_file)


# Now that we have understood how to generate audio embeddings,
# we need to create the data for our Siamese network.
# As we know, Siamese network accepts the data in a pair,
# we define the function for getting our data. We will
# create the genuine pair as (Dogs, Sub_dogs) and assign
# label as 1 and imposite pair as (Dogs, Cats) and assign label as 0.

# In[7]:
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


def get_training_data():
    
    pairs = []
    labels = []
    
    Dogs = glob.glob('data/audio/Dogs/*.wav')
    Sub_dogs = glob.glob('data/audio/Sub_dogs/*.wav')
    Cats = glob.glob('data/audio/Cats/*.wav')
    
    
    np.random.shuffle(Sub_dogs)
    np.random.shuffle(Cats)
    
    for i in range(min(len(Cats),len(Sub_dogs))):
        #imposite pair
        if (i % 2) == 0:
            pairs.append([audio2vector(Dogs[randint(0,3)]),audio2vector(Cats[i])])
            labels.append(0)
            
        #genuine pair
        else:
            pairs.append([audio2vector(Dogs[randint(0,3)]),audio2vector(Sub_dogs[i])])
            labels.append(1)
            
            
    return np.array(pairs), np.array(labels)


# In[8]:

df_preprocessed, features, target_feature = load_data()
df_preprocessed = df_preprocessed.dropna(subset=['target_binary_intrusion'], how='any')

X_train, X_test, y_train, y_test = train_test_split(df_preprocessed[features], df_preprocessed['target_binary_intrusion'],
                                      test_size=0.15, stratify=df_preprocessed['target_binary_intrusion'])


# Now that we have successfully generated our data, we build our Siamese network. We define our base network which is used for feature extraction, we use three dense layers with dropout layer in between.

# In[10]:


def build_base_network(input_shape):
    input = Input(shape=input_shape)
    x = Dense(128, activation='relu')(input)
    x = Dropout(0.1)(x)
    # x = Dense(128, activation='relu')(x)
    #     # x = Dropout(0.1)(x)
    #     # x = Dense(128, activation='relu')(x)
    return Model(input, x)


# Next, we feed the audio pair to the base network, which will return the features:

# In[11]:


input_dim = X_train.shape[1]

audio_a = Input(shape=(input_dim,))
audio_b = Input(shape=(input_dim,))


# In[12]:


base_network = build_base_network((input_dim,))

feat_vecs_a = base_network(audio_a)
feat_vecs_b = base_network(audio_b)


# These feat_vecs_a and feat_vecs_b are the feature vectors of our audio pair. Next,
# we feed this feature vectors to the energy function to compute a distance between them, we use Euclidean distance as our energy function:

# In[13]:


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


# In[14]:


distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([feat_vecs_a, feat_vecs_b])


# 
# Next, we set the epoch length to 13 and we use RMS prop for optimization.

# In[15]:


epochs = 13
rms = RMSprop()


# In[16]:


model = Model(input=[audio_a, audio_b], output=distance)


# Lastly, we define our loss function as contrastive_loss  and compile the model.

# In[17]:


def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


# In[18]:


model.compile(loss=contrastive_loss, optimizer=rms)


# Now, we train our model,

# In[19]:

#
# audio_1 = X_train[:, 0]
# audio_2 = X_train[:, 1]

x_train, x_test = cv_preprocessing(X_train, X_test)

# In[20]:


model.fit(X_train, y_train, validation_split=.25,
          batch_size=128, verbose=2, nb_epoch=epochs, class_weight={0: 1, 1: 5})

y_pred = model.predict_proba(X_test)[:, 1]
print_results(y_test, y_pred)
