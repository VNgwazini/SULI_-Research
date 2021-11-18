#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 10:28:17 2019

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
import matplotlib.pyplot as plt

model = KNeighborsClassifier(n_neighbors=5)
index = 0   

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

    
         bin_3_dec = -600#-450
         bin_2_dec = -450#-300
         bin_1_dec = -300#-150
         bin_1_inc = 250# 150
         bin_2_inc = 450# 300
         bin_3_inc = 600# 450

    
        
    
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
             

#         data_stats = data[eachDF]['Change'].describe()
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

    # load data
#    series = Series.H
    # prepare data
#    train_size = int(len(H[index]))
#    print('Train-Size',train_size)

#    best_HeatK = 1
#    best_HeatScore = 0.0
#    kValue = 1
#    best_Y_pred = ...
#    for eachModel in range(0,100):
#        kValue=kValue + 2
#        


#        print('kValue is: [',kValue,']')
#    print('X+1     :',np.reshape(H[index][start+1:end+1],(1,-1)))
#    print('Y+1     :',np.reshape(B[index][start+1:end+1],(1,-1)))
#        print('Current Window: \n',current_window)
#        print('Y: \n',B[index])
#        Z = np.reshape(?[index],(-1,1))

#        #Split data into train and test sets
#        X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.75,test_size=0.25,shuffle=False)
#
#         #Create Model
#        model = KNeighborsClassifier(n_neighbors=kValue)
#         
#         #Train Model
#        model.fit(X_train,Y_train.ravel())
##        print(model)
#         
#         #Predict Output
#        Y_pred = model.predict(X_test)
##        print('Prediction: ',Y_pred)
#        knn_HeatScore = float(metrics.accuracy_score(Y_test,Y_pred))
##        print('\n===============================================================================\n')
##        print('\nKNN Accuracy: ',metrics.accuracy_score(Y_test,Y_pred))
##        print('\n===============================================================================\n')
##        print('knn_HeatScore is: [',knn_HeatScore,']')
#        if(knn_HeatScore > best_HeatScore):
#            best_HeatScore = knn_HeatScore
#            best_HeatK =  kValue
#            best_Y_pred = Y_pred
    start,end = -1,128        #Create Model
    count = 0
    correct,wrong = 0,0
    accuracyPred = []
    for item in range(0,len(H[index])- 129): 
        start,end = start + 1, end + 1
#        print(start,' ',end)
        #restore to [start:end] for sliding window instead of growing window
        X = H[index][0:end]
        Y = B[index][0:end]        
  
#        for prediction in H[index]:
        count = count + 1
            
        X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1,shuffle=False)    
 
             #Train Model
        model.fit(X_train,Y_train.ravel())
             #print(model)
        
             #Predict Output
        best_Y_pred = model.predict(X_test[0:1])
#        print('Y Raw   :',np.reshape(Y,(1,-1)))
#        print('Y Trian :',np.reshape(Y_train,(1,-1)))
#        print('Y Test  :',np.reshape(Y_test,(1,-1)))
#        print('Y Pred  :',np.reshape(best_Y_pred,(1,-1)))
#        print('\n===============================================================================\n')
#        print('Real Value:',Y_test[0:1],'| Pred Value:',best_Y_pred,'| KNN Accuracy: ',metrics.accuracy_score(Y_test[0:1],best_Y_pred[0:1]))
#        print('\n===============================================================================\n')
        if(best_Y_pred == Y_test[0:1]):
            correct = correct + 1
            accuracyPred.append(1)
        elif(best_Y_pred != Y_test[0:1]):
            wrong = wrong + 1
            accuracyPred.append(-1)
    print('Correct : ',correct)
    print('Wrong   : ',wrong)
    X = H[index]
    Y = B[index]        
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1,shuffle=False)    
    Y_pred = model.predict(X_test)

    print('\n===============================================================================\n')
    print(confusion_matrix(Y_test,Y_pred))
    print('\n===============================================================================\n')
    print(classification_report(Y_test,Y_pred))
    
    print('\n===============================================================================\n')
     #Comapre Train and Test scores fort KNN
    print('Train Score: ',model.score(X_train, Y_train))
    print('Test Score : ',model.score(X_test,Y_test))
    print('\n===============================================================================\n')
    
    plt.plot(accuracyPred[0:64])
    plt.show()
    
     #histgram
    dataFrames[0]['Heat_Release'].hist()
    plt.suptitle('Label Frequencies: E058', fontsize=16)
    plt.grid(True)
    plt.show()

#    mybinsDF = pd.DataFrame(myBins)
#    print('Bin size',len(mybinsDF[0][0]),'\nCount',count)
#     #histgram
#    mybinsDF.hist()
#    plt.suptitle('Label Frequencies: E058', fontsize=16)
#    plt.grid(True)
#    plt.show()
    
     #compare all three signals on line graph
    f,(ax1,ax2,ax3,ax4,ax5) = plt.subplots(5, sharex=True,sharey=True)
    ax1.plot(dataFrames[0]['Heat_Release'],c='red',label='E058',
                  alpha=0.7)
    ax1.set_title('Heat Release Signal Comparison: E058 -> E075', fontsize=16)
    ax2.plot(dataFrames[1]['Heat_Release'],c='orange',label='E60',
                  alpha=0.7)
    ax3.plot(dataFrames[2]['Heat_Release'],c='purple',label='E65-1',
                  alpha=0.7)
    ax4.plot(dataFrames[3]['Heat_Release'],c='green',label='E065-2',
                  alpha=0.7)
    ax5.plot(dataFrames[4]['Heat_Release'],c='blue',label='E075',
                  alpha=0.7)
    
    f.subplots_adjust(hspace=0)
    #plt.setp([ax.get_ticklabels() for ax in f.axes[:-1]],visible=False)
    f.legend(loc=5)
    plt.xlabel("Cycles (T)",fontsize=12)
    plt.ylabel("Amplitudes",fontsize=12)
    plt.show()
    

    
    
    
   
#        print('\nBest KNN Accuracy: ',metrics.accuracy_score(Y_test[0:1],best_Y_pred[0:1]))
#        print('\n===============================================================================\n')

##    start = 0
##    end = 128
#    tscv = TimeSeriesSplit(n_splits=2)
#    print('TSCV: \n',tscv)  
#    TimeSeriesSplit(max_train_size=None, n_splits=2)
##    for eachWindow in range(0,len(H)):
###        H_window = np.reshape(H[index][start:end],(-1,1))
###        print('\nH Window is: \n',H_window)
###        B_window = np.reshape(B[index][start:end],(-1,1))
###        print('\nB Window is: \n',B_window)
#    mySplit = list(tscv.split(H[index]))
#    print('\n\nmySplit: \n\n', mySplit)
# 
#    for train_index, test_index in tscv.split(H[index]):
#    for item in range(0,len(H[index])):
##        print("TRAIN:", train_index, "TEST:", test_index)
#        #Create Model
#        start,end = 0,8
#        xWin = []
#        yWin = []
##        window_size = 8
#        for i in range(start,end):
#            xWin.clear()
#            yWin.clear()
#            xWin.append(H[index][i])
#            yWin.append(B[index][i])
#        X_train, X_test = np.reshape(xWin,(-1,1)), np.reshape(H[index][end],(-1,1))
#        Y_train, Y_test = np.reshape(yWin,(-1,1)), np.reshape(B[index][end],(-1,1))
#        model = KNeighborsClassifier(n_neighbors=best_HeatK)
##            H_window = np.reshape(H[index][train_index:test_index],(-1,1))
##            B_window = np.reshape(B[index][train_index:test_index],(-1,1))
#
#        #Train Model
#        model.fit(X_train,Y_train.ravel())
#        #print(model)
#        #Predict Output
#        
#        Y_pred = model.predict(X_test)
##        print('\nX test: \n', Y_test)
##        print('\nY pred: \n', Y_pred)
#        print('\n===============================================================================\n')
#        print('\nKNN Accuracy: ',metrics.accuracy_score(Y_test,Y_pred))
#        print('\n===============================================================================\n')
#        if (start < (len(H[index]) -end -8)):
#            start = start + 1
#        if (end < (len(H[index])-end)):
#            end = end + 1
#
 
#            
#        X_train,X_test,Y_train,Y_test = train_test_split(H_window,B_window,test_size=0.33,stratify=B_window)
##        print('X Test: \n',X_test)
#        
#         #Create Model
#        model = KNeighborsClassifier(n_neighbors=best_HeatK)
##         
#         #Train Model
#        model.fit(X_train,Y_train.ravel())
#         #print(model)
#         
#             #Predict Output
#        best_Y_pred = model.predict(X_test)
#        print('\n===============================================================================\n')
#        print('\nKNN Accuracy: ',metrics.accuracy_score(Y_test,best_Y_pred))
#        print('\n===============================================================================\n')
#        if (start < (len(H[index]) -end -8)):
#            start = start + 1
#        if (end < (len(H[index])-end)):
#            end = end + 1
#    

            
#    print('\n===============================================================================\n')
#    print('Best Heat K-Value: ',best_HeatK,' with accuracy of ',best_HeatScore)
#    print('\n===============================================================================\n')
#    print(best_Y_pred)
#    print('\n===============================================================================\n')
#    print('\nKNN Accuracy: ',metrics.accuracy_score(Y_test,best_Y_pred))
#    print('\n===============================================================================\n')
#    print(confusion_matrix(Y_test,best_Y_pred))
#    print('\n===============================================================================\n')
#    print(classification_report(Y_test,best_Y_pred))
#    
#    print('\n===============================================================================\n')
#    #Comapre Train and Test scores fort KNN
#    print('Train Score: ',model.score(X_train, Y_train))
#    print('Test Score : ',model.score(X_test,Y_test))
#    print('\n===============================================================================\n')



    
#========================================================================================================================================================================================
#                MAIN FUNCTION
#========================================================================================================================================================================================

def main():
#    print('\nIn main: \n')
    #print(EngineData)
    CsvToDF(EngineData,fileIndex)

    
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from collections import namedtuple


    n_groups = 6
    
    sliding = (42.4,56.2,59.0,60.4,60.6,58.2)
    growing = (57.7,57.7,57.2,55.4,54.0,52.3)
    

    fig, ax = plt.subplots()

    index = np.arange(n_groups)
    bar_width = 0.35

    opacity = 0.7
    error_config = {'ecolor': '0.3'}

    rects1 = ax.bar(index, sliding, bar_width,
                alpha=opacity, color='green',
                error_kw=error_config,
                label='Sliding')

    rects2 = ax.bar(index + bar_width, growing, bar_width,
                alpha=opacity, color='blue',
                error_kw=error_config,
                label='Growing')
    

    ax.set_xlabel('Training Window Size')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Accuracy by Training Window Size and Method')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(('8 Cycles', '16 Cycles', '32 Cycles', '64 Cycles', '96 Cycles', '128 Cycles'))
    ax.legend(loc=1)

    fig.tight_layout()
    plt.show()


#========================================================================================================================================================================================
#                MAIN FUNCTION
#========================================================================================================================================================================================

    
main()
    