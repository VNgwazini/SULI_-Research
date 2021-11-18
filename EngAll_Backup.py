#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 10:06:10 2019

@author: 1vn
"""

#==========================================================================
# Load libraries
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
#import plotly.plotly as py

import numpy as np

from sklearn import  metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


#==========================================================================
#==========================================================================
e058     ='Engine/EngineData/Processed_Engine_Data/e058_Headers_OFF.csv'
e058_H   ='Engine/EngineData/Processed_Engine_Data/e058_Headers_On.csv'

e060     ='Engine/EngineData/Processed_Engine_Data/e060_Headers_OFF.csv'
e060_H   ='Engine/EngineData/Processed_Engine_Data/e060_Headers_On.csv'

e065_1   ='Engine/EngineData/Processed_Engine_Data/e065_1_Headers_OFF.csv'
e065_1_H ='Engine/EngineData/Processed_Engine_Data/e065_1_Headers_On.csv'
 
e065_2   ='Engine/EngineData/Processed_Engine_Data/e065_2_Headers_OFF.csv'
e065_2_H ='Engine/EngineData/Processed_Engine_Data/e065_2_Headers_On.csv'

e075     ='Engine/EngineData/Processed_Engine_Data/e075_Headers_OFF.csv'
e075_H   ='Engine/EngineData/Processed_Engine_Data/e075_Headers_On.csv'
# 
# 
# 
# 
datasets_Labels = [e058_H,e060_H,e065_1_H,e065_2_H,e075_H]
datasets_No_Labels = [e058,e060,e065_1,e065_2,e075]
col_names=['Heat_Release','DOWN_OR_UP','Cnsctv_N_Y','Percentile']


#print('DataSets Labels: ',datasets_Labels[0][1])

EngineData = [datasets_Labels,datasets_No_Labels]
#DataStats = [datasets_Labels, datasets_No_Labels.decribe] 

dataFrames = []
# 
 # =============================================================================
 #APPEND CSV FILES TO DATA FRAME
 
 #this loop is for keeping existing column names        
for eachDataset in range (0,int(len(EngineData)/2)):
    for eachSubSet in range (0,len(EngineData[0])):
         #print(EngineData[eachDataset][eachSubSet])
         dfH = pd.read_csv((EngineData[eachDataset][eachSubSet]))
         dataFrames.append(dfH)
#         print(dataFrames[eachSubSet].head(5))        

# =============================================================================
# #this loop is for adding column names
# for eachDataset in range ((int(len(EngineData)/2)),int(len(EngineData))):
#     for eachSubSet in range (0,len(EngineData[1])):
#         #print(EngineData[eachDataset][eachSubSet])
#         dfNH = pd.read_csv((EngineData[eachDataset][eachSubSet]),names = col_names ,header='infer')
#         dataFrames.append(dfNH)
#         #print(dataFrames[eachSubSet].tail(5))
# =============================================================================



#===================================================================================
 #MODIFY CSV WITH NEW DATA FRAME EDITS
fileIndex = 0 
for eachDF in range (0,int(len(dataFrames))):
     #print(dataFrames[eachDF].head(5))
 #    dataFrames[eachDF].to_csv(dataFrames[eachDF])
 #    dataFrames[eachDF].to_csv(EngineData[1][eachDF])
     
#     print(dataFrames[eachDF])
 
     #run some statistics on the data
     data_stats = dataFrames[eachDF].describe()
#     print('Data Description: \n',data_stats)        
     
     #min
     minAmp = data_stats.Heat_Release[3]
     #25%
     #print('minAmp: ',minAmp)
     lower25 = data_stats.Heat_Release[4]
     #50%
     medianAmp =data_stats.Heat_Release[5]
     #75%
     upper75 = data_stats.Heat_Release[6]
     #max
     maxAmp = data_stats.Heat_Release[7]
     
     dataFrames[eachDF]['DOWN_OR_UP'].at[0] = int(1);
     dataFrames[eachDF]['Cnsctv_N_Y'].at[0] = int(0);
     dataFrames[eachDF]['Percentile'].at[0] = int(1)
     
     for eachRow in range(1,len(dataFrames[0]['Heat_Release'])):
         
         #=========================DOWN_OR_UP============================================
         if(dataFrames[eachDF]['Heat_Release'].at[eachRow] < dataFrames[eachDF]['Heat_Release'].at[eachRow-1]):
             #print('\nrow value: ',dataFrames[eachDF]['Percentile'][eachRow])
             #print('1')
             dataFrames[eachDF]['DOWN_OR_UP'].at[eachRow]= int(0)
             
             if(dataFrames[eachDF]['DOWN_OR_UP'].at[eachRow] != dataFrames[eachDF]['DOWN_OR_UP'].at[eachRow-1]):
                     #print('0')
                     dataFrames[eachDF]['Cnsctv_N_Y'].at[eachRow]= int(0)
             
             elif(dataFrames[eachDF]['DOWN_OR_UP'].at[eachRow] == dataFrames[eachDF]['DOWN_OR_UP'].at[eachRow-1]):
                    #print('2')     
                    dataFrames[eachDF]['Cnsctv_N_Y'].at[eachRow]= int(1)
                    #print('row value: ',dataFrames[eachDF]['Percentile'].at[eachRow])
             
         elif(dataFrames[eachDF]['Heat_Release'].at[eachRow] > dataFrames[eachDF]['Heat_Release'].at[eachRow-1]):
             #print('\nrow value: ',dataFrames[eachDF]['Percentile'].at[eachRow])
             #print('2')     
             dataFrames[eachDF]['DOWN_OR_UP'].at[eachRow]= int(1)
             #print('row value: ',dataFrames[eachDF]['Percentile'].at[eachRow])
             
             if(dataFrames[eachDF]['DOWN_OR_UP'].at[eachRow] != dataFrames[eachDF]['DOWN_OR_UP'].at[eachRow-1]):
                     #print('0')
                     dataFrames[eachDF]['Cnsctv_N_Y'].at[eachRow]= int(0)
             
             elif(dataFrames[eachDF]['DOWN_OR_UP'].at[eachRow] == dataFrames[eachDF]['DOWN_OR_UP'].at[eachRow-1]):
                    #print('2')     
                    dataFrames[eachDF]['Cnsctv_N_Y'].at[eachRow]= int(1)
                    #print('row value: ',dataFrames[eachDF]['Percentile'].at[eachRow])
                    
         #=========================DOWN_OR_UP============================================
         
         
#
#
#     for eachRow in range(1,len(dataFrames[0]['Heat_Release'])):
#       
#         
#         #=========================DOWN_OR_UP============================================
#         if(dataFrames[eachDF]['DOWN_OR_UP'].at[eachRow] != dataFrames[eachDF]['DOWN_OR_UP'].at[eachRow-1]):
#             #print('0')
#             dataFrames[eachDF]['Cnsctv_N_Y'].at[eachRow]= int(0)
#             
#         elif(dataFrames[eachDF]['DOWN_OR_UP'].at[eachRow] == dataFrames[eachDF]['DOWN_OR_UP'].at[eachRow-1]):
#             #print('2')     
#             dataFrames[eachDF]['Cnsctv_N_Y'].at[eachRow]= int(1)
#         #=========================DOWN_OR_UP============================================
#         

     for eachRow in range(0,len(dataFrames[0]['Heat_Release'])):
         #=========================PERCENTILE============================================
         if(dataFrames[eachDF]['Heat_Release'].at[eachRow] <= lower25):
             #print('\nrow value: ',dataFrames[eachDF]['Percentile'][eachRow])
             #print('1')
             dataFrames[eachDF]['Percentile'].at[eachRow]= int(1)
             #print('row value: ',dataFrames[eachDF]['Percentile'].at[eachRow])
             
         elif(dataFrames[eachDF]['Heat_Release'].at[eachRow] > lower25 and dataFrames[eachDF]['Heat_Release'].at[eachRow] <= medianAmp):
             #print('\nrow value: ',dataFrames[eachDF]['Percentile'].at[eachRow])
             #print('2')     
             dataFrames[eachDF]['Percentile'].at[eachRow]= int(2)
             #print('row value: ',dataFrames[eachDF]['Percentile'].at[eachRow])
                  
         elif(dataFrames[eachDF]['Heat_Release'].at[eachRow] > medianAmp and dataFrames[eachDF]['Heat_Release'].at[eachRow] <= upper75):
             #print('\nrow value: ',dataFrames[eachDF]['Percentile'].at[eachRow])
             #print('3')     
             dataFrames[eachDF]['Percentile'].at[eachRow]= int(3)
             #print('row value: ',dataFrames[eachDF]['Percentile'].at[eachRow])
                  
         elif(dataFrames[eachDF]['Heat_Release'].at[eachRow] > upper75):
             #print('\nrow value:    ',dataFrames[eachDF]['Percentile'].at[eachRow])
             #print('4')     
             dataFrames[eachDF]['Percentile'].at[eachRow]= int(4)
             #print('row value:    ',dataFrames[eachDF]['Percentile'].at[eachRow])
         #=========================PERCENTILE============================================

             
             
             
             
             
             
     #print(dataFrames[eachDF].head(5)) 
          #dataFrames[eachDF]['Percentile'] += newPercentile
     #print('Percentile: \n',dataFrames[eachDF]['Percentile'][1000:])       
     dataFrames[eachDF].to_csv(EngineData[1][fileIndex],index=False)
#     print(EngineData[1][fileIndex]) 
     fileIndex = fileIndex+1
#     print('File Index is now: [',fileIndex,']')
#     print(dataFrames[eachDF])
#     print('Percentile: \n',dataFrames[eachDF]['Percentile'])
 
#     #dataFrames[eachDF]['Percentile'] += newPercentile
#     print('Percentile: \n',dataFrames[eachDF]['Percentile'][1000:])
#=============================================================================

Heats = []
UpDowns = []
Cnsctvs = []
Percs = []

for eachDF in range (0,int(len(dataFrames))):
#    print(dataFrames[eachDF])
    #run some statistics on the data
    data_stats = dataFrames[eachDF].describe()
#     print('Data Description: \n',data_stats)        
     
    #min
    minAmp = data_stats.Heat_Release[3]
    #25%
    #print('minAmp: ',minAmp)
    lower25 = data_stats.Heat_Release[4]
    #50%
    medianAmp =data_stats.Heat_Release[5]
    #75%
    upper75 = data_stats.Heat_Release[6]
    #max
    maxAmp = data_stats.Heat_Release[7]
     
    Heat_Release_List = dataFrames[eachDF]['Heat_Release'].tolist()
    print('heat realse list: \n',Heat_Release_List )
    Heats.append(np.reshape((np.asarray(Heat_Release_List)),(-1,1)))
    
    UpDowns_List = dataFrames[eachDF]['DOWN_OR_UP'].tolist()
    UpDowns.append(np.reshape((np.asarray(UpDowns_List)),(-1,1)))
    
    Cnsctvs_List = dataFrames[eachDF]['Cnsctv_N_Y'].tolist()
    Cnsctvs.append(np.reshape((np.asarray(Cnsctvs_List)),(-1,1)))
    
    Percentile_List = dataFrames[eachDF]['Percentile'].tolist()
    Percs.append(np.reshape((np.asarray(Percentile_List)),(-1,1)))
    
#    Heat_Release_e060 = dataFrames[eachDF]['Heat_Release'].tolist()
#    Heat_Release_e065_1 = dataFrames[eachDF]['Heat_Release'].tolist()
#    Heat_Release_e065_2 = dataFrames[eachDF]['Heat_Release'].tolist()
#    Heat_Release_e075 = dataFrames[eachDF]['Heat_Release'].tolist()
     
#    Heat_Release_e058 = dataFrames[eachDF]['Heat_Release'].tolist()
#    Percentile_e058 = dataFrames[eachDF]['Percentile'].tolist()
#    Heat_Release_e060 = dataFrames[eachDF]['Heat_Release'].tolist()
#    Heat_Release_e065_1 = dataFrames[eachDF]['Heat_Release'].tolist()
#    Heat_Release_e065_2 = dataFrames[eachDF]['Heat_Release'].tolist()
#    Heat_Release_e075 = dataFrames[eachDF]['Heat_Release'].tolist()
    
    #Diff_Test_List = dataFrames[eachDF]['Difference'].tolist()
    
# =============================================================================
#build and test KNN model

#K NEAREST NEIGHBOR CLASSIFICATION ---HEAT RELEASE
    
    
 #Features
#chaotic signal
HtRs = Heats[0]
#print('\nHtrS: ',HtRs)
UpDs = UpDowns[0]
#print('\nHtrS: ',UpDs)
CnSc = Cnsctvs[0]
PrCs = Percs[0]


HtRs4 = Heats[4]
#print('\nHtrS: ',HtRs)
UpDs4 = UpDowns[4]
#print('\nHtrS: ',UpDs)
CnSc4 = Cnsctvs[4]
PrCs4 = Percs[4]






X = np.reshape([[HtRs],[UpDs]],(-1,1))

#print('X shape',X.shape)
#print('X: \n',X)
#clean signal
Z = np.reshape([[HtRs4],[UpDs4]],(-1,1))
#print('Z shape',X.shape)
#print(X)Heat_Release_list
 
 #Label
Y = np.reshape([[PrCs],[CnSc]],(-1,1))
#print('Y shape',X.shape)
#print(Y)
 
# =============================================================================

       
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
    if(knn_HeatScore > best_HeatScore):
        best_HeatScore = knn_HeatScore
        best_HeatK =  kValue
        #print('kValue is now: [',best_HeatK,']')

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
        #print('kValue is now: [',best_TestK,']')
print('\n===============================================================================\n')
print('Best Heat K-Value: ',best_HeatK,' with accuracy of ',best_HeatScore)
print('Best Test K-Value: ',best_TestK,' with accuracy of ',knn_TestScore)
print('\n===============================================================================\n')
#============================================================================================
#============================================================================================

#Split data into train and test sets
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3)
 
 #Create Model
model = KNeighborsClassifier(n_neighbors=best_HeatK)
 
 #Train Model
model.fit(X_train,Y_train)
 #print(model)
 
 #Predict Output
Y_pred = model.predict(X_test)
print (Y_pred)
print('\nKNN Accuracy: ',metrics.accuracy_score(Y_test,Y_pred))
print(confusion_matrix(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))
 #Comapre Train and Test scores fort KNN
print('Train Score: ',model.score(X_train, Y_train))
print('Test Score: ',model.score(X_test,Y_test))

print('\n===============================================================================\n')
 
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


