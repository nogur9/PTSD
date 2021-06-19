#!/usr/bin/env python
# coding: utf-8

# In[53]:


import numpy as np
from load_data import load_data
from preprocessing import cv_preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score


# In[54]:


def sample_points(k):
    x = np.random.rand(k,50)
    y = np.random.choice([0, 1], size=k, p=[.5, .5]).reshape([-1,1])
    return x,y


# In[55]:


x, y = sample_points(10)
print(x[0])
print(y[0])


class MetaSGD(object):
    def __init__(self):
        
        #initialize number of tasks i.e number of tasks we need in each batch of tasks
        self.num_tasks = 2
        
        #number of samples i.e number of shots  -number of data points (k) we need to have in each task
        self.num_samples = 10

        #number of epochs i.e training iterations
        self.epochs = 10000
        
        #hyperparameter for the inner loop (inner gradient update)
        self.alpha = 0.0001
        
        #hyperparameter for the outer loop (outer gradient update) i.e meta optimization
        self.beta = 0.0001
       
        #randomly initialize our model parameter theta
        self.theta = np.random.normal(size=85).reshape(85, 1)
         
        #randomly initialize alpha with same shape as theta
        self.alpha = np.random.normal(size=85).reshape(85, 1)
      
    #define our sigmoid activation function  
    def sigmoid(self,a):
        return 1.0 / (1 + np.exp(-a))
    
    
    #now let us get to the interesting part i.e training :P
    def train(self, XTrain, YTrain, XTest, YTest):
        YTrain = YTrain.values.reshape(-1, 1)
        YTest = YTest.values.reshape(-1, 1)
        #for the number of epochs,
        for e in range(self.epochs):        
            
            self.theta_ = []
            
            #for task i in batch of tasks
            for i in range(self.num_tasks):
               
                #sample k data points and prepare our train set
                
                a = np.matmul(XTrain, self.theta)

                YHat = self.sigmoid(a)

                #since we are performing classification, we use cross entropy loss as our loss function
                print("first", np.matmul(-YTrain.T, np.log(YHat)))
                print("second", np.matmul((1 -YTrain.T), np.log(1 - YHat)))
                print("third", self.num_samples)

                loss = ((np.matmul(-YTrain.T, np.log(YHat)) - np.matmul((1 -YTrain.T), np.log(1 - YHat)))/self.num_samples)
                print("loss", loss.values[0])
                loss = loss.values[0]
                #minimize the loss by calculating gradients
                gradient = np.matmul(XTrain.T, (YHat - YTrain)) / self.num_samples
                print(f"gradient={gradient.shape}")

                #update the gradients and find the optimal parameter theta' for each of tasks
                print("self.theta", self.theta)
                print("self.alpha", self.alpha, self.alpha.shape)
                print("np.multiply(self.alpha,gradient)", np.multiply(self.alpha,gradient))
                self.theta_.append(self.theta - (np.multiply(self.alpha,gradient)))
                
     
            #initialize meta gradients
            meta_gradient = np.zeros(self.theta.shape)
                        
            for i in range(self.num_tasks):
            
                #sample k data points and prepare our test set for meta training
                

                #predict the value of y
                print("XTest", XTest.shape)
                print("self.theta_[i]", self.theta_[i].shape)
                a = np.matmul(XTest, self.theta_[i])
                print("a", a)
                YPred = self.sigmoid(a)
                           
                #compute meta gradients
                meta_gradient += np.matmul(XTest.T, (YPred - YTest)) / self.num_samples

            #update our randomly initialized model parameter theta with the meta gradients
            self.theta = self.theta-self.beta*meta_gradient/self.num_tasks
                       
            #update our randomly initialized hyperparameter alpha with the meta gradients
            self.alpha = self.alpha-self.beta*meta_gradient/self.num_tasks
                                       
            if e%1000==0:
                print("Epoch {}: Loss {}\n".format(e,loss))
                print('Updated Model Parameter Theta\n')
                print('Sampling Next Batch of Tasks \n')
                print('---------------------------------\n')


# In[61]:



df_preprocessed, features, target_feature = load_data()
df_preprocessed = df_preprocessed.dropna(subset=['target_binary_intrusion'], how='any')

X_train, X_test, y_train, y_test = train_test_split(df_preprocessed[features], df_preprocessed['target_binary_intrusion'],
                                      test_size=0.15, stratify=df_preprocessed['target_binary_intrusion'])
X_train, X_test = cv_preprocessing(X_train, X_test)

model = MetaSGD()
model.train(X_train, y_train, X_test, y_test)

