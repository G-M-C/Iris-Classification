#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
from tensorflow import keras





import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')





df = pd.read_csv('iris_data_set.csv')




display(df)





df = df[df.species!='virginica']





df.species.unique()





sns.pairplot(df,hue='species')





from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.preprocessing import LabelEncoder





X = df.drop('species',axis=1)
y = df['species'].values
encoder = LabelEncoder()
val = encoder.fit_transform(y)
y = pd.get_dummies(val).values





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)





import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import random
random.set_seed(7)
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop





model = Sequential()
model.add(Dense(4,input_shape=(4,),activation='relu'))
model.add(Dense(2,activation='softmax'))
optimizer = RMSprop()
model.compile(optimizer,loss='binary_crossentropy',metrics=['accuracy'])
model.summary()





model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=1)





predictions = model.predict(X_test)




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






sns.heatmap((confusion_matrix(y_test_label,predictions_label)),annot=True,cmap="YlGnBu")




print(classification_report(y_test_label,predictions_label))





acc = (accuracy_score(y_test,predictions))
print('{}%'.format(acc*100))





"""for i in range(0,len(y_test_label)):
    print('{} : {}'.format(y_test_label[i],predictions_label[i]))"""







