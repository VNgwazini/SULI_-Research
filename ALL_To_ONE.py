#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 10:06:10 2019

@author: 1vn
"""

# Load libraries
import pandas as pd
#import plotly.plotly as py

import numpy as np

from sklearn import  metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
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
col_names=['Heat_Release','Change','DOWN_OR_UP','Cnsctv_N_Y','Bin']


#print('DataSets Labels: ',datasets_Labels[0][1])

EngineData = [datasets_Labels,datasets_No_Labels]
#print(EngineData)
#DataStats = [datasets_Labels, datasets_No_Labels.decribe] 
 
dataFrames = []
#print('Variable: ',dataFrames)
fileIndex = 0
#print('Variable: ',fileIndex)  




#========================================================================================================================================================================================
#========================================================================================================================================================================================




def CsvToDF(data,F_index ):

#    print('\nIn USA: \n')
#    print('\nDataFrames: \n',dataFrames)
    for eachDataset in range (0,int(len(data)/2)):
        for eachSubSet in range (0,len(data[0])):
#             print(data[eachDataset][eachSubSet])
             dfH = pd.read_csv((data[eachDataset][eachSubSet]))
             dataFrames.append(dfH)
    

    return DFToCSV(dataFrames, F_index)



##========================================================================================================================================================================================
#========================================================================================================================================================================================




 #MODIFY CSV WITH NEW DATA FRAME EDITS

#pass dataframes
def DFToCSV(data, F_index):
#    print('Data',data)
#    print('\nIn DFToCSV: \n')

    for eachDF in range (0,int(len(data))):
#         print(data[eachDF].head(5))
     #    data[eachDF].to_csv(data[eachDF])
     #    data[eachDF].to_csv(EngineData[1][eachDF])
         
    #     print(data[eachDF])
     
         #run some statistics on the data
#         data_stats = data[eachDF].describe()
#         print('Data Description: \n',data_stats)        
         
# =============================================================================
#          #min
#          minAmp = data_stats.Heat_Release[3]
# #         #25%
# #         #print('minAmp: ',minAmp)
# #         lower25 = data_stats.Heat_Release[4]
# #         #50%
# #         medianAmp =data_stats.Heat_Release[5]
# #         #75%
# #         upper75 = data_stats.Heat_Release[6]
#          #max
#          maxAmp = data_stats.Heat_Release[7]
# 
#          
# =============================================================================
         
         
         data[eachDF]['DOWN_OR_UP'].at[0] = int(1);
         data[eachDF]['Cnsctv_N_Y'].at[0] = int(0);
         data[eachDF]['Change'].at[0] = int(0)
         
         
         for eachRow in range(1,len(data[0]['Heat_Release'])):
             
             #=========================Change============================================
             data[eachDF]['Change'].at[eachRow]= int(data[eachDF]['Heat_Release'].at[eachRow] - data[eachDF]['Heat_Release'].at[eachRow-1])
                     #print('row value: ',data[eachDF]['Bin'].at[eachRow])
         
         
         for eachRow in range(1,len(data[0]['Heat_Release'])):
             
             #=========================DOWN_OR_UP============================================
             if(data[eachDF]['Heat_Release'].at[eachRow] < data[eachDF]['Heat_Release'].at[eachRow-1]):
                     #print('\nrow value: ',data[eachDF]['Bin'][eachRow])
                     #print('1')
                     data[eachDF]['DOWN_OR_UP'].at[eachRow]= int(0)
                     
                     if(data[eachDF]['DOWN_OR_UP'].at[eachRow] != data[eachDF]['DOWN_OR_UP'].at[eachRow-1]):
                             #print('0')
                             data[eachDF]['Cnsctv_N_Y'].at[eachRow]= int(0)
                     
                     elif(data[eachDF]['DOWN_OR_UP'].at[eachRow] == data[eachDF]['DOWN_OR_UP'].at[eachRow-1]):
                            #print('2')     
                            data[eachDF]['Cnsctv_N_Y'].at[eachRow]= int(1)
                            #print('row value: ',data[eachDF]['Bin'].at[eachRow])
                 
             elif(data[eachDF]['Heat_Release'].at[eachRow] > data[eachDF]['Heat_Release'].at[eachRow-1]):
                     #print('\nrow value: ',data[eachDF]['Bin'].at[eachRow])
                     #print('2')     
                     data[eachDF]['DOWN_OR_UP'].at[eachRow]= int(1)
                     #print('row value: ',data[eachDF]['Bin'].at[eachRow])
                     
                     if(data[eachDF]['DOWN_OR_UP'].at[eachRow] != data[eachDF]['DOWN_OR_UP'].at[eachRow-1]):
                             #print('0')
                             data[eachDF]['Cnsctv_N_Y'].at[eachRow]= int(0)
                     
                     elif(data[eachDF]['DOWN_OR_UP'].at[eachRow] == data[eachDF]['DOWN_OR_UP'].at[eachRow-1]):
                            #print('2')     
                            data[eachDF]['Cnsctv_N_Y'].at[eachRow]= int(1)
                            #print('row value: ',data[eachDF]['Bin'].at[eachRow])

    
         bin_3_dec = -450
         bin_2_dec = -300
         bin_1_dec = -150
         bin_1_inc =  150
         bin_2_inc =  300
         bin_3_inc =  450

    
        
    
         for eachRow in range(0,len(data[0]['Change'])):
             #=========================Bin============================================
             if(data[eachDF]['Change'].at[eachRow] <= bin_3_dec):
#                 print('\nrow value: ',data[eachDF]['Change'][eachRow])
#                 print('-4')
                 data[eachDF]['Bin'].at[eachRow]= int(-4)
                 #print('row value: ',data[eachDF]['Bin'].at[eachRow])
                 
             elif(data[eachDF]['Change'].at[eachRow] > bin_3_dec and data[eachDF]['Change'].at[eachRow] <= bin_2_dec):
#                 print('\nrow value: ',data[eachDF]['Change'].at[eachRow])
#                 print('-3')     
                 data[eachDF]['Bin'].at[eachRow]= int(-3)
                 #print('row value: ',data[eachDF]['Bin'].at[eachRow])
                      
             elif(data[eachDF]['Change'].at[eachRow] > bin_2_dec and data[eachDF]['Change'].at[eachRow] <= bin_1_dec):
#                 print('\nrow value: ',data[eachDF]['Change'].at[eachRow])
#                 print('-2')     
                 data[eachDF]['Bin'].at[eachRow]= int(-2)
                 #print('row value: ',data[eachDF]['Bin'].at[eachRow])
                 
             elif(data[eachDF]['Change'].at[eachRow] > bin_1_dec and data[eachDF]['Change'].at[eachRow] <= int(0)):
#                 print('\nrow value: ',data[eachDF]['Change'].at[eachRow])
#                 print('-1')     
                 data[eachDF]['Bin'].at[eachRow]= int(-1)
                 #print('row value: ',data[eachDF]['Bin'].at[eachRow])
                      
             elif(data[eachDF]['Change'].at[eachRow] > int(0) and data[eachDF]['Change'].at[eachRow] <= bin_1_inc):
#                 print('\nrow value: ',data[eachDF]['Change'].at[eachRow])
#                 print('1')     
                 data[eachDF]['Bin'].at[eachRow]= int(1)
                 #print('row value: ',data[eachDF]['Bin'].at[eachRow])
             
             elif(data[eachDF]['Change'].at[eachRow] > bin_1_inc and data[eachDF]['Change'].at[eachRow] <= bin_2_inc):
#                 print('\nrow value: ',data[eachDF]['Change'].at[eachRow])
#                 print('2')     
                 data[eachDF]['Bin'].at[eachRow]= int(2)
                 #print('row value: ',data[eachDF]['Bin'].at[eachRow])
                      
             elif(data[eachDF]['Change'].at[eachRow] > bin_2_inc and data[eachDF]['Change'].at[eachRow] <= bin_3_inc):
#                 print('\nrow value: ',data[eachDF]['Change'].at[eachRow])
#                 print('3')     
                 data[eachDF]['Bin'].at[eachRow]= int(3)
                 #print('row value: ',data[eachDF]['Bin'].at[eachRow])

             elif(data[eachDF]['Change'].at[eachRow] > bin_3_inc ):
#                 print('\nrow value: ',data[eachDF]['Change'].at[eachRow])
#                 print('4')     
                 data[eachDF]['Bin'].at[eachRow]= int(4)
                 #print('row value: ',data[eachDF]['Bin'].at[eachRow])
             #=========================Bin============================================
             

         data_stats = data[eachDF]['Change'].describe()
#         print('Data Description: \n',data_stats)


         data[eachDF].to_csv(EngineData[1][F_index],index=False)
#         print(EngineData[1][fileIndex]) 
         F_index = F_index+1
#         print('File Index is now: [',fileIndex,']')


    return colsToList(data)




#========================================================================================================================================================================================
#========================================================================================================================================================================================




def colsToList(data):
    Heats = []
    UpDowns = []
    Cnsctvs = []
    Bins = []

#    print('\nIn colsToList: \n')
    eachDF = None
    
    for eachDF in range (0,int(len(data))):
#        print(data[eachDF].head(5))

#        print('\nIn For Loop: \n')
        #    print(data[eachDF])
#   # 
#   #     #run some statistics on the data
#        data_stats = data[eachDF].describe()
        #print('Data Description: \n',data_stats)        
#   #      
#   #     #min
#   #     minAmp = data_stats.Heat_Release[3]
#   #     #25%
#   #     #print('minAmp: ',minAmp)
#   #     lower25 = data_stats.Heat_Release[4]
#   #     #50%
#   #     medianAmp =data_stats.Heat_Release[5]
#   #     #75%
#   #     upper75 = data_stats.Heat_Release[6]
#   #     #max
#   #     maxAmp = data_stats.Heat_Release[7]
    #     
        Heat_Release_List = data[eachDF]['Heat_Release'].tolist()
        Heats.append(np.reshape((np.asarray(Heat_Release_List)),(-1,1)))
        #print('heat realse list: \n',Heat_Release_List )
    #    
        UpDowns_List = data[eachDF]['DOWN_OR_UP'].tolist()
        UpDowns.append(np.reshape((np.asarray(UpDowns_List)),(-1,1)))
    #    
        Cnsctvs_List = data[eachDF]['Cnsctv_N_Y'].tolist()
        Cnsctvs.append(np.reshape((np.asarray(Cnsctvs_List)),(-1,1)))
    #    
        Bin_List = data[eachDF]['Bin'].tolist()
        Bins.append(np.reshape((np.asarray(Bin_List)),(-1,1)))
    #    
    #    Heat_Release_e060 = data[eachDF]['Heat_Release'].tolist()
    #    Heat_Release_e065_1 = data[eachDF]['Heat_Release'].tolist()
    #    Heat_Release_e065_2 = data[eachDF]['Heat_Release'].tolist()
    #    Heat_Release_e075 = data[eachDF]['Heat_Release'].tolist()
         
    #    Heat_Release_e058 = data[eachDF]['Heat_Release'].tolist()
    #    Bin_e058 = data[eachDF]['Bin'].tolist()
    #    Heat_Release_e060 = data[eachDF]['Heat_Release'].tolist()
    #    Heat_Release_e065_1 = data[eachDF]['Heat_Release'].tolist()
    #    Heat_Release_e065_2 = data[eachDF]['Heat_Release'].tolist()
    #    Heat_Release_e075 = data[eachDF]['Heat_Release'].tolist()
        
    #    #Diff_Test_List = data[eachDF]['Difference'].tolist()
    #    
    #
    #print('Heats: \n',Heats)
    if eachDF is None:
        raise ValueError("Empty data iterable: {!r:100}".format(data))    
    return set_XYZ(Heats, UpDowns, Cnsctvs, Bins)




#========================================================================================================================================================================================
#========================================================================================================================================================================================




def set_XYZ(H,U,C,B):   
# =============================================================================
# #    print('\nIn set_XYZ: \n')
#     #chaotic signal
#     HtRs = H[0]
#     #print('\nHtrS: ',HtRs)
#     UpDs = U[0]
#     #print('\nHtrS: ',UpDs)
#     CnSc = C[0]
#     PrCs = B[0]
#     
#     
#     HtRs4 = H[4]
#     #print('\nHtrS: ',HtRs)
#     UpDs4 = U[4]
#     #print('\nHtrS: ',UpDs)
# #    CnSc4 = myCnsctvs[4]
# #    PrCs4 = myBins[4]
# =============================================================================
    
    
    First3_Heats = []
    First3_Heats.append(H[0])
#    print('\nFirst 3 Heats: \n',First3_Heats)
#    print('\nFirst 3 Heats Size: \n',len(First3_Heats))
    
    First3_UpDowns = []
    First3_UpDowns.append(U[0])
#    print('\nFirst3_UpDowns: \n',First3_UpDowns)
#    print('\nFirst3_UpDowns Size: \n',len(First3_UpDowns))
    
    First3_Cnsctvs = []
    First3_Cnsctvs.append(C[0])
#    print('\nFirst3_Cnsctvs: \n',First3_Cnsctvs)
#    print('\nFirst3_Cnsctvs Size: \n',len(First3_Cnsctvs))
    
    First3_Bins = []
    First3_Bins.append(B[0])
#    print('\nFirst3_Bins: \n',First3_Bins)
#    print('\nFirst3_Bins Size: \n',len(First3_Bins))
    
    Last2_Heats = []
    Last2_Heats.append(H[1])
#    print('\nLast 2 Heats: \n',Last2_Heats)
    
    Last2_UpDowns = []
    Last2_UpDowns.append(U[1])
#    print('\nLast2_UpDowns: \n',Last2_UpDowns)
    
    Last2_Cnsctvs = []
    Last2_Cnsctvs.append(C[1])
#    print('\nLast2_Cnsctvs: \n',Last2_Cnsctvs)
    
    Last2_Bins = []
    Last2_Bins.append(B[1])
#    print('\nLast2_Bins: \n',Last2_Bins)

    
    

    
    

    
    X = np.reshape([First3_Heats],(-1,1))
    
#    print('X shape',X.shape)
#    print('X: \n',X)
    #clean signal
    Z = np.reshape([Last2_Heats],(-1,1))
    #print('Z shape',X.shape)
    #print(X)Heat_Release_list
     
     #Label
    Y = np.reshape([First3_Bins],(-1,1))
    #print('Y shape',X.shape)
    #print(Y)
        
    return findBestK(X, Y, Z)




#========================================================================================================================================================================================
#========================================================================================================================================================================================




def findBestK(feature_X,label_Y,feature_Z):
#    print('\nIn findBestK: \n')

    #Split data into train and test sets
    X_train,X_test,Y_train,Y_test = train_test_split(feature_X,label_Y,test_size=0.3,stratify=label_Y)
    
    
    best_HeatK = 1
    best_TestK = 1
    best_HeatScore = 0.0
    best_TestScore = 0.0
    kValue = 1
    for eachModel in range(1,100):
        kValue=kValue + 2
#        print('kValue is: [',kValue,']')
         #Create Model
        model = KNeighborsClassifier(n_neighbors=kValue)
         
         #Train Model
        model.fit(X_train,Y_train.ravel())
#        print(model)
         
         #Predict Output
        Y_pred = model.predict(X_test)
        knn_HeatScore = float(metrics.accuracy_score(Y_test,Y_pred))
#        print('knn_HeatScore is: [',knn_HeatScore,']')
#        print(knn_HeatScore,' > ',best_HeatScore,' ?')
        if(knn_HeatScore > best_HeatScore):
            best_HeatScore = knn_HeatScore
            best_HeatK =  kValue
#            print('kValue is now: [',best_HeatK,']')
    
    #============================================================================================
    #============================================================================================
         
         #Split data into train and test sets
        Z_train,Z_test,Y_train,Y_test = train_test_split(feature_Z,label_Y,test_size=0.3,stratify=label_Y)
         
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
    
    return finalKNN(best_HeatK,feature_X,label_Y,feature_Z)




#========================================================================================================================================================================================
#========================================================================================================================================================================================





def finalKNN(best_K,feature_X,label_Y,feature_Z):
#    print('\nIn finalKNN: \n')

    #Split data into train and test sets
    X_train,X_test,Y_train,Y_test = train_test_split(feature_X,label_Y,test_size=0.3,stratify=label_Y)    
    
     #Create Model
    model = KNeighborsClassifier(n_neighbors=best_K)
     
     #Train Model
    model.fit(X_train,Y_train.ravel())
     #print(model)
     
     #Predict Output
    Y_pred = model.predict(X_test)
    print(Y_pred)
    print('\n===============================================================================\n')
    print('\nKNN Accuracy: ',metrics.accuracy_score(Y_test,Y_pred))
    print('\n===============================================================================\n')
    print(confusion_matrix(Y_test,Y_pred))
    print('\n===============================================================================\n')
    print(classification_report(Y_test,Y_pred))
    
    print('\n===============================================================================\n')
     #Comapre Train and Test scores fort KNN
    print('Train Score: ',model.score(X_train, Y_train))
    print('Test Score : ',model.score(X_test,Y_test))
    print('\n===============================================================================\n')

    #Split data into train and test sets
    Z_train,Z_test,Y_train,Y_test = train_test_split(feature_Z,label_Y,test_size=0.3,stratify=label_Y)
        
     #Predict Output
    Y_pred = model.predict(Z_test)
    print(Y_pred)
    print('\nKNN Accuracy: ',metrics.accuracy_score(Y_test,Y_pred))
    print(confusion_matrix(Y_test,Y_pred))
    print(classification_report(Y_test,Y_pred))
    
     #Comapre Train and Test scores fort KNN
    print('Test Score: ',model.score(Z_test,Y_test))
    
#========================================================================================================================================================================================
#                MAIN FUNCTION
#========================================================================================================================================================================================

def main():
#    print('\nIn main: \n')
    #print(EngineData)
    CsvToDF(EngineData,fileIndex)


#========================================================================================================================================================================================
#                MAIN FUNCTION
#========================================================================================================================================================================================

    
main()
    
    
    
    