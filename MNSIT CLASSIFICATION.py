#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.image as mpimg


# In[2]:


import matplotlib.pyplot as plt


# In[3]:


import tensorflow as tf


# In[ ]:


#Loading the dataset


# In[4]:


(X_train,y_train),(X_test,y_test)=tf.keras.datasets.mnist.load_data()


# In[6]:


#pixel represenation of a number present in X_train[0]
print(X_train[0])


# In[7]:


#So it was the pixel reprsentaion of 5
print(y_train[0])


# In[8]:


#Getting the shape of image
X_train.shape
#we have 28*28 image where we have 60000 images in t=our training set


# In[9]:


len(X_train[0][0])


# In[10]:


len(X_train[0])


# In[12]:


plt.imshow(X_train[0])


# In[15]:


plt.imshow(X_train[1],cmap='gist_gray')


# In[17]:


y_train[1]


# In[ ]:


#Using Deep Learning CNN to process  our data


# In[18]:


import keras


# In[19]:


X_train=X_train.reshape(X_train.shape[0],28,28,1)
X_test=X_test.reshape(X_test.shape[0],28,28,1)
#reshaping them into 4d tensors so they can be processed using keras


# In[20]:


X_train.shape


# In[21]:


X_test.shape


# In[22]:


from keras.layers import Conv2D,Dropout
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.models import Sequential
from keras.layers import Dense


# In[23]:


model=Sequential()


# In[24]:


model.add(Conv2D(32,kernel_size=(3,3),input_shape=(28,28,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(16,kernel_size=(2,2),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(1,1)))
model.add(Flatten())
model.add(Dense(units=128,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=10,activation='softmax'))


# In[ ]:





# In[25]:


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[33]:


from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test=to_categorical(y_test
                     )


# In[34]:


y_train.shape


# In[36]:


y_train


# In[35]:


model.fit(X_train,y_train)


# In[31]:


# i have got approx 95% accuracy in only one epoch


# In[37]:


model.evaluate(X_test,y_test)


# In[ ]:


#testing


# In[38]:


#prediction
pred=model.predict_proba(X_test[1].reshape(1,28,28,1))
pred


# In[ ]:


num_class=model.predict_classes(X_test[1].reshape(1,28,28,1))


# In[41]:


print(num_class)


# In[42]:


plt.imshow(X_test[1].reshape(28,28))


# In[43]:


pred1=model.predict_classes(X_test[1009].reshape(1,28,28,1))


# In[44]:


pred1


# In[46]:


plt.imshow(X_test[1009].reshape(28,28))


# In[47]:


#hence we can see that we have coorectly predicted our output


# In[ ]:




