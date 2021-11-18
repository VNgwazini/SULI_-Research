#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 08:58:05 2019

@author: 1vn
"""

import pandas as pd

#import plotly.plotly as py

import numpy as np

from sklearn import  metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier






#load dataset 'e058-u-001-h.csv'
filename1 = 'Engine/EngineData/e058-No-Cycles.csv'
filename2 = 'Engine/EngineData/e058-No-Cycles-No-Labels.csv'
filename3 = 'Engine/EngineData/e075-u-001-h.csv'
filename4 = 'Engine/EngineData/Processed_Engine_Data/e058_Diff.csv'
filename5 = 'Engine/EngineData/Processed_Engine_Data/e075_Diff.csv'
col_names=['IMEP_Net','IMEP_Gross','Heat_Release','Change','Percentile']
dataset_Labels = pd.read_csv(filename1)#,names = col_names)
dataset_No_Labels = pd.read_csv(filename2,names = col_names)
dataset_Test = pd.read_csv(filename3)
dataset_Diff = pd.read_csv(filename4)
dataset_DiffTest = pd.read_csv(filename5)





Heat_Release_list = dataset_Labels['Heat_Release'].tolist()
Change_list = dataset_Labels['Change'].tolist()
Percentile_list = dataset_Labels['Percentile'].tolist()
Test_Data_List = dataset_Test['HR'].tolist()
Diff_Data_List = dataset_Diff['Difference'].tolist()
Diff_Change_List = dataset_Diff['Change'].tolist()
Diff_Test_List = dataset_DiffTest['Difference'].tolist()

print('\n=================================================================\n')


#build and test KNN model

#K NEAREST NEIGHBOR CLASSIFICATION ---HEAT RELEASE
 
 #Features
#chaotic signal
X = np.reshape((np.asarray(Diff_Data_List)),(-1,1))
#clean signal
Z = np.reshape((np.asarray(Diff_Test_List)),(-1,1))
#print(X)Heat_Release_list
 
 #Label
Y = np.reshape((np.asarray(Diff_Change_List)),(-1,1))
#print(Y)
 
 #Split data into train and test sets
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3)
 
 #Create Model
model = KNeighborsClassifier(n_neighbors=1)
 
 #Train Model
model.fit(X_train,Y_train.ravel())
 #print(model)
 
 #Predict Output
Y_pred = model.predict(X_test)
print (Y_pred)
print('KNN Accuracy: ',metrics.accuracy_score(Y_test,Y_pred))
print(confusion_matrix(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))
 #Comapre Train and Test scores fort KNN
print('Train Score: ',model.score(X_train, Y_train))
print('Test Score: ',model.score(X_test,Y_test))

print('\n=================================================================\n')
 
 #Split data into train and test sets
Z_train,Z_test,Y_train,Y_test = train_test_split(Z,Y,test_size=0.3)
 
 #Predict Output
Y_pred = model.predict(Z_test)
print (Y_pred)
print('\nKNN Accuracy: ',metrics.accuracy_score(Y_test,Y_pred))
print(confusion_matrix(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))

 #Comapre Train and Test scores fort KNN
print('Test Score: ',model.score(Z_test,Y_test))

# using classifier chains
from skmultilearn.problem_transform import ClassifierChain
from sklearn.naive_bayes import GaussianNB

# initialize classifier chains multi-label classifier
# with a gaussian naive bayes base classifier
classifier = ClassifierChain(GaussianNB())

# train
classifier.fit(X_train, Y_train)

# predict
predictions = classifier.predict(X_test)

accuracy_score(Y_test,predictions)