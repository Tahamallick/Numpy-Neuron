#!/usr/bin/env python
# coding: utf-8

# # Implement Neural Network (or Logistic Regression) From Scratch

# Predicting if a person would buy life insurnace based on his age using logistic regression
# 

# Above is a binary logistic regression problem as there are only two possible outcomes (i.e. if person buys insurance or he/she doesn't).
# 
# 

# In[1]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv(r"C:\Users\Taha\Desktop\main\insurance_data.csv")
df.head()


# Split train and test set
# 
# 

# In[3]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df[['age','affordibility']],df.bought_insurance,test_size=0.2, random_state=25)


# Preprocessing: Scale the data so that both age and affordibility are in same scaling range
# 
# 

# In[4]:


X_train_scaled = X_train.copy()
X_train_scaled['age'] = X_train_scaled['age'] / 100

X_test_scaled = X_test.copy()
X_test_scaled['age'] = X_test_scaled['age'] / 100


# Model Building: First build a model in keras/tensorflow and see what weights and bias values it comes up with. We will than try to reproduce same weights and bias in our plain python implementation of gradient descent. Below is the architecture of our simple neural network

# In[5]:


model = keras.Sequential([
    keras.layers.Dense(1, input_shape=(2,), activation='sigmoid', kernel_initializer='ones', bias_initializer='zeros')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_scaled, y_train, epochs=500)


# Evaluate the model on test set
# 
# 

# In[6]:


model.evaluate(X_test_scaled,y_test)


# In[7]:


model.predict(X_test_scaled)


# In[8]:


y_test


# Now get the value of weights and bias from the model
# 
# 

# In[9]:


coef, intercept = model.get_weights()


# In[10]:


coef, intercept


# This means w1=0.77850705, w2=0.6977987, bias =-0.39703578
# 
# 

# In[11]:


def sigmoid(x):
        import math
        return 1 / (1 + math.exp(-x))
sigmoid(18)


# In[12]:


X_test


# Instead of model.predict, write our own prediction function that uses w1,w2 and bias
# 
# 

# In[13]:


def prediction_function(age, affordibility):
    weighted_sum = coef[0]*age + coef[1]*affordibility + intercept
    return sigmoid(weighted_sum)

prediction_function(.47, 1)


# In[14]:


prediction_function(.18, 1)


# Now we start implementing gradient descent in plain python. Again the goal is to come up with same w1, w2 and bias that keras model calculated. We want to show how keras/tensorflow would have computed these values internally using gradient descent
# 
# First write couple of helper routines such as sigmoid and log_loss

# In[15]:


def sigmoid_numpy(X):
   return 1/(1+np.exp(-X))

sigmoid_numpy(np.array([12,0,1]))


# In[16]:


def log_loss(y_true, y_predicted):
    epsilon = 1e-15
    y_predicted_new = [max(i,epsilon) for i in y_predicted]
    y_predicted_new = [min(i,1-epsilon) for i in y_predicted_new]
    y_predicted_new = np.array(y_predicted_new)
    return -np.mean(y_true*np.log(y_predicted_new)+(1-y_true)*np.log(1-y_predicted_new))


# All right now comes the time to implement our own custom neural network class !! yay !!!
# 
# 

# In[17]:


class myNN:
    def __init__(self):
        self.w1 = 1 
        self.w2 = 1
        self.bias = 0
        
    def fit(self, X, y, epochs, loss_thresold):
        self.w1, self.w2, self.bias = self.gradient_descent(X['age'],X['affordibility'],y, epochs, loss_thresold)
        print(f"Final weights and bias: w1: {self.w1}, w2: {self.w2}, bias: {self.bias}")
        
    def predict(self, X_test):
        weighted_sum = self.w1*X_test['age'] + self.w2*X_test['affordibility'] + self.bias
        return sigmoid_numpy(weighted_sum)

    def gradient_descent(self, age,affordability, y_true, epochs, loss_thresold):
        w1 = w2 = 1
        bias = 0
        rate = 0.5
        n = len(age)
        for i in range(epochs):
            weighted_sum = w1 * age + w2 * affordability + bias
            y_predicted = sigmoid_numpy(weighted_sum)
            loss = log_loss(y_true, y_predicted)
            
            w1d = (1/n)*np.dot(np.transpose(age),(y_predicted-y_true)) 
            w2d = (1/n)*np.dot(np.transpose(affordability),(y_predicted-y_true)) 

            bias_d = np.mean(y_predicted-y_true)
            w1 = w1 - rate * w1d
            w2 = w2 - rate * w2d
            bias = bias - rate * bias_d
            
            if i%50==0:
                print (f'Epoch:{i}, w1:{w1}, w2:{w2}, bias:{bias}, loss:{loss}')
            
            if loss<=loss_thresold:
                print (f'Epoch:{i}, w1:{w1}, w2:{w2}, bias:{bias}, loss:{loss}')
                break

        return w1, w2, bias


# In[18]:


customModel = myNN()
customModel.fit(X_train_scaled, y_train, epochs=8000, loss_thresold=0.4631)


# In[19]:


coef, intercept


# This shows that in the end we were able to come up with same value of w1,w2 and bias using a plain python implementation of gradient descent function

# In[20]:


X_test_scaled


# (1) Predict using custom model
# 
# 

# In[21]:


customModel.predict(X_test_scaled)


# (2) Predict using tensorflow model
# 
# 

# In[22]:


model.predict(X_test_scaled)


# In[ ]:




