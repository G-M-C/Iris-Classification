#!/usr/bin/env python
# coding: utf-8

# # Soft Computing Assignment - 2

# ## A neural network capable of classifying the Iris flowering plant into 'setosa' and 'versicolor' species

# ### Author : George M Cherian , CS5A , 28

# In[77]:


import numpy as np
import pandas as pd
from tensorflow import keras


# In[78]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[79]:


df = pd.read_csv('iris_data_set.csv')


# In[80]:


display(df)


# In[81]:


#Dropping the Iris-virginica species
df = df[df.species!='virginica']


# In[82]:


df.species.unique()


# ##### Visualizing Data

# In[83]:


sns.pairplot(df,hue='species')


# ##### Importing Scikit-Learn

# ##### Training - Testing & Splitting Data 

# In[84]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.preprocessing import LabelEncoder


# ##### One Hot Encoding

# In[85]:


X = df.drop('species',axis=1)
y = df['species'].values
encoder = LabelEncoder()
val = encoder.fit_transform(y)
y = pd.get_dummies(val).values


# In[86]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[87]:


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import random
#random.set_seed(7)
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop


# ##### Building the Neural Network

# In[88]:


model = Sequential()
model.add(Dense(4,input_shape=(4,),activation='relu'))
model.add(Dense(2,activation='softmax'))
optimizer = RMSprop()
model.compile(optimizer,loss='binary_crossentropy',metrics=['accuracy'])
model.summary()


# In[89]:


model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=1)


# In[90]:


predictions = model.predict(X_test)


# In[91]:


y_test = np.argmax(y_test,axis=1)
predictions = np.argmax(predictions,axis=1)

y_test_label = []
for val in y_test:
    if val == 0:
        y_test_label.append('setosa')
    else:
        y_test_label.append('versicolor')


predictions_label = []
for val in predictions:
    if val == 0:
        predictions_label.append('setosa')
    else:
        predictions_label.append('versicolor')




# ##### Evaluation Metrics

# In[92]:


sns.heatmap((confusion_matrix(y_test_label,predictions_label)),annot=True,cmap="YlGnBu")


# In[93]:


print(classification_report(y_test_label,predictions_label))


# In[94]:


acc = (accuracy_score(y_test,predictions))
print('{}%'.format(acc*100))


# In[95]:


for i in range(0,len(y_test_label)):
    print('{} : {}'.format(y_test_label[i],predictions_label[i]))


# In[ ]:




