#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 09:54:42 2019

@author: 1vn
"""

import pandas as pd

#import plotly.plotly as py

import numpy as np

from sklearn import  metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#============================================================================================
#============================================================================================


#load dataset 'e058-u-001-h.csv'
filename1 = 'Engine/EngineData/e058-No-Cycles.csv'
filename2 = 'Engine/EngineData/e058-No-Cycles-No-Labels.csv'
filename3 = 'Engine/EngineData/e075-u-001-h.csv'
col_names=['IMEP_Net','IMEP_Gross','Heat_Release','Change','Percentile']
dataset_Labels = pd.read_csv(filename1)#,names = col_names)
dataset_No_Labels = pd.read_csv(filename2,names = col_names)
dataset_Test = pd.read_csv(filename3)

#============================================================================================
#============================================================================================


Heat_Release_list = dataset_Labels['Heat_Release'].tolist()
Change_list = dataset_Labels['Change'].tolist()
Percentile_list = dataset_Labels['Percentile'].tolist()
Test_Data_List = dataset_Test['HR'].tolist()

print('\n=================================================================\n')
#build and test KNN model

#K NEAREST NEIGHBOR CLASSIFICATION ---HEAT RELEASE
 
 #Features
X = np.reshape((np.asarray(Heat_Release_list)),(-1,1))
Z = np.reshape((np.asarray(Test_Data_List)),(-1,1))
#print(X)Heat_Release_list
 
 #Label
Y = np.reshape((np.asarray(Change_list)),(-1,1))
#print(Y)
 
 #Split data into train and test sets
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3)

best_HeatK = 1
best_TestK = 1
best_HeatScore = 0.0
best_TestScore = 0.0
kValue = 0
for eachModel in range(1,100):
    kValue=kValue + 1
#    print('kValue is: [',kValue,']')
     #Create Model
    model = KNeighborsClassifier(n_neighbors=kValue)
     
     #Train Model
    model.fit(X_train,Y_train.ravel())
     #print(model)
     
     #Predict Output
    Y_pred = model.predict(X_test)
    knn_HeatScore = float(metrics.accuracy_score(Y_test,Y_pred))
#    print('knn_HeatScore is: [',knn_HeatScore,']')
    print(knn_HeatScore,' > ',best_HeatScore,' ?')
    if(knn_HeatScore > best_HeatScore):
        best_HeatScore = knn_HeatScore
        print('best_HeatScore is now: ',best_HeatScore)
        best_HeatK =  kValue
        print('kValue is now: [',best_HeatK,']')

#============================================================================================
#============================================================================================
     
     #Split data into train and test sets
    Z_train,Z_test,Y_train,Y_test = train_test_split(Z,Y,test_size=0.3)
     
     #Predict Output
    Y_pred = model.predict(Z_test)

    
    knn_TestScore = float(metrics.accuracy_score(Y_test,Y_pred))
#    print('knn_TestScore is: [',knn_TestScore,']')
    if(knn_TestScore > best_TestScore):
        best_TestScore = knn_TestScore
        best_TestK =  kValue
        print('kValue is now: [',best_TestK,']')

print('Best Heat K-Value: ',best_HeatK,' with accuracy of ',best_HeatScore)
print('Best Test K-Value: ',best_TestK,' with accuracy of ',knn_TestScore)