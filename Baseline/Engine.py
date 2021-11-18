#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 14:43:10 2019

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

#load dataset 'e058-u-001-h.csv'
filename1 = 'Engine/EngineData/e058-No-Cycles.csv'
filename2 = 'Engine/EngineData/e058-No-Cycles-No-Labels.csv'
filename3 = 'Engine/EngineData/e075-u-001-h.csv'
col_names=['IMEP_Net','IMEP_Gross','Heat_Release','Change','Percentile']
dataset_Labels = pd.read_csv(filename1)#,names = col_names)
dataset_No_Labels = pd.read_csv(filename2,names = col_names)
dataset_Test = pd.read_csv(filename3)


# =============================================================================
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
col_names=['Heat_Release','Change','Percentile']

df_Labels = []
#for each
df_No_Labels = []

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
         df = pd.read_csv((EngineData[eachDataset][eachSubSet]))
         dataFrames.append(df)
         #run some statistics on the data
         data_stats = dataFrames[eachSubSet].describe()
         print('Data Description: \n',data_stats)        

        #print(dataFrames[eachSubSet].head(5))        

#this loop is for adding column names
for eachDataset in range ((int(len(EngineData)/2)),int(len(EngineData))):
    for eachSubSet in range (0,len(EngineData[1])):
        #print(EngineData[eachDataset][eachSubSet])
        df = pd.read_csv((EngineData[eachDataset][eachSubSet]),names = col_names)
        dataFrames.append(df)
        #print(dataFrames[eachSubSet].tail(5))

 
# =============================================================================
#Here is the order of the data frames in the list
        
#0 Engine/EngineData/Processed_Engine_Data/e058_Headers_On.csv
#1 Engine/EngineData/Processed_Engine_Data/e060_Headers_On.csv
#2 Engine/EngineData/Processed_Engine_Data/e065_1_Headers_On.csv
#3 Engine/EngineData/Processed_Engine_Data/e065_2_Headers_On.csv
#4 Engine/EngineData/Processed_Engine_Data/e075_Headers_On.csv
#------------------------------------------------------------------------------
#5 Engine/EngineData/Processed_Engine_Data/e058_Headers_OFF.csv
#6 Engine/EngineData/Processed_Engine_Data/e060_Headers_OFF.csv
#7 Engine/EngineData/Processed_Engine_Data/e065_1_Headers_OFF.csv
#8 Engine/EngineData/Processed_Engine_Data/e065_2_Headers_OFF.csv
#9 Engine/EngineData/Processed_Engine_Data/e075_Headers_OFF.csv        
# =============================================================================         
    
 #print('DataFrames Length: \n',len(dataFrames))
 #print('DataFrames:1 \n',dataFrames[0])
 #print('DataFrames:6 \n',dataFrames[5])
 #loop through label and no label
 # =============================================================================
# # for eachDataset in range (0,len(EngineData)):
# #     for eachSubSet in range (0,len(EngineData[0])):
# #         print(EngineData[eachDataset][eachSubSet])
# #         df = to_csv((EngineData[eachDataset][eachSubSet]),names = col_names)
# #         dataFrames.to_csv((EngineData[eachDataset][eachSubSet]),names = col_names)
# #         print(dataFrames[eachSubSet])
# #===================================================================================
# #MODIFY CSV WITH NEW DATA FRAME EDITS
# for eachDF in range (5,len(dataFrames)):
#     #print(dataFrames[eachDF].head(5))
# #    dataFrames[eachDF].to_csv(dataFrames[eachDF])
# #    dataFrames[eachDF].to_csv(EngineData[1][eachDF])
#     
#     #run some statistics on the data
#    data_stats = dataFrames[eachDF].describe()
#    print('Data Description: \n',data_stats)        
#     
#     #min
#     minAmp = data_stats.Heat_Release[3]
#     #25%
#     #print('minAmp: ',minAmp)
#     lower25 = data_stats.Heat_Release[4]
#     #50%
#     medianAmp =data_stats.Heat_Release[5]
#     #75%
#     upper75 = data_stats.Heat_Release[6]
#     #max
#     maxAmp = data_stats.Heat_Release[7]
#     
#     for eachRow in range(0,len(dataFrames[5]['Heat_Release'])):
#         print('Currently on row: ',eachRow,' viewing: ',dataFrames[5]['Heat_Release'][eachRow])
#         
#         
#         if(dataFrames[eachDF]['Heat_Release'].at[eachRow] <= lower25):
#             #print('\nrow value: ',dataFrames[eachDF]['Percentile'][eachRow])
#             print('1')
#             dataFrames[eachDF]['Percentile'].at[eachRow]= 1
#             #print('row value: ',dataFrames[eachDF]['Percentile'].at[eachRow])
#             
#         elif(dataFrames[eachDF]['Heat_Release'].at[eachRow] > lower25 and dataFrames[eachDF]['Heat_Release'].at[eachRow] <= medianAmp):
#             #print('\nrow value: ',dataFrames[eachDF]['Percentile'].at[eachRow])
#             print('2')     
#             dataFrames[eachDF]['Percentile'].at[eachRow]= 2
#             #print('row value: ',dataFrames[eachDF]['Percentile'].at[eachRow])
#                  
#         elif(dataFrames[eachDF]['Heat_Release'].at[eachRow] > medianAmp and dataFrames[eachDF]['Heat_Release'].at[eachRow] <= upper75):
#             #print('\nrow value: ',dataFrames[eachDF]['Percentile'].at[eachRow])
#             print('3')     
#             dataFrames[eachDF]['Percentile'].at[eachRow]= 3
#             #print('row value: ',dataFrames[eachDF]['Percentile'].at[eachRow])
#                  
#         elif(dataFrames[eachDF]['Heat_Release'].at[eachRow] > upper75):
#             #print('\nrow value:    ',dataFrames[eachDF]['Percentile'].at[eachRow])
#             print('4')     
#             dataFrames[eachDF]['Percentile'].at[eachRow]= 4
#             #print('row value:    ',dataFrames[eachDF]['Percentile'].at[eachRow])
#     dataFrames[eachDF].to_csv(dataFrames[eachDF],columns = col_names, index=False, sep=',')
#     #print('Percentile: \n',dataFrames[eachDF]['Percentile'])
# 
#     #dataFrames[eachDF]['Percentile'] += newPercentile
#     print('Percentile: \n',dataFrames[eachDF]['Percentile'][1000:])
# =============================================================================
    
# =============================================================================

# =============================================================================
# #create a list that determines what percentile each value is in
# print(Heat_Release_list)
# print(Heat_Release_list)
# print(len(Heat_Release_list))
# percentile = []
# print('Percentile: ',percentile)
# 
# for eachElement in range (len(Heat_Release_list)):
#     if(Heat_Release_list[eachElement] <= lower25):
#         percentile.append(1)
#     elif((Heat_Release_list[eachElement] > lower25) & (Heat_Release_list[eachElement] <= medianAmp)):
#         percentile.append(2)        
#     elif((Heat_Release_list[eachElement] > medianAmp) & (Heat_Release_list[eachElement] <= upper75)):
#         percentile.append(3)
#     elif(Heat_Release_list[eachElement] > upper75):
#         percentile.append(4)
#     print(percentile[eachElement])
# print(len(percentile))
# print('\nPercentile Ranges Encoded: \n',percentile) 
# =============================================================================
#print('DataFrames: ',dataFrames)
# =============================================================================
 
 #print('DATASETS 00: ',datasets[0]['Heat_Release'][:10])
 #print('LENGTH: ',len(datasets))
 
 
 
 #values = dataset.loc[0:1008,col_names[0]:col_names[2]]
values = pd.read_csv(filename1,header=None)
 #print('Values: \n',values)
 
 #Datashape
print('Data Shape: \n',dataset_Labels.shape)
 #each file has 1009 instances of 3 attributes
 
 #peak at the first 20 rows
print('Data Head: \n',dataset_Labels.head(20))
 
 #run some statistics on the data
data_stats = dataset_Labels.describe()
print('Data Description: \n',data_stats)
 
 ##analyze the distribution by showing how many elements are in each category
 #print(dataset.groupby(names).size())
IMEP_Net_list = dataset_Labels['IMEP_Net'].tolist()
IMEP_Gross_list = dataset_Labels['IMEP_Gross'].tolist()
Heat_Release_list = dataset_Labels['Heat_Release'].tolist()
Change_list = dataset_Labels['Change'].tolist()
Percentile_list = dataset_Labels['Percentile'].tolist()
Test_Data_List = dataset_Test['HR'].tolist()
dataset_Labels_List = list(zip(IMEP_Net_list,IMEP_Gross_list,Heat_Release_list,Change_list,Percentile_list))
# 
# #=============================================================================
# #==========================================================================
 
 #3x3 scatter matrix
import seaborn as sns
sns.set(style="ticks")
plot = sns.pairplot(dataset_No_Labels)
plt.grid(True)
plot.fig.suptitle('Signal Scatter Matrix (Seaborn): E058', fontsize=16)
# 
# #========================================================================
# #==========================================================================
 
 #visualize dataset
 
 #box plot
dataset_No_Labels.plot(kind='box', subplots=True, layout=(5,1), sharex=True, sharey=False)
plt.suptitle('Signal Box Plot: E058', fontsize=16)
plt.grid(True)
plt.show()

 #histgram
dataset_No_Labels.hist()
plt.suptitle('Signal Histogram: E058', fontsize=16)
plt.grid(True)
plt.show()
 
 #scatter plot matrix
scatter_matrix(dataset_No_Labels)
plt.suptitle('Signal Scatter Matrix: E058', fontsize=16)
plt.grid(True)
plt.show()
 
 #compare all three signals on line graph
f,(ax1,ax2,ax3,ax4,ax5) = plt.subplots(5, sharex=True,sharey=False)
ax1.plot(IMEP_Net_list[:20])
ax1.set_title('Signal Comparison: E058', fontsize=16)
ax2.plot(IMEP_Gross_list[:20])
ax3.plot(Heat_Release_list[:20])
ax4.plot(Change_list[:20])
ax5.plot(Percentile_list[:20],color='r')
f.subplots_adjust(hspace=0)
#plt.setp([ax.get_ticklabels() for ax in f.axes[:-1]],visible=False)
plt.grid(True)
plt.show()
 
# # =============================================================================
# #==========================================================================
 
 #K NEAREST NEIGHBOR CLASSIFICATION ---HEAT RELEASE
 
 #Features
X = np.reshape((np.asarray(Test_Data_List)),(-1,1))
Z = np.reshape((np.asarray(Heat_Release_list)),(-1,1))
#print(X)Heat_Release_list
 
 #Label
Y = np.reshape((np.asarray(Change_list)),(-1,1))
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
print('KNN Accuracy: ',metrics.accuracy_score(Y_test,Y_pred))
print(confusion_matrix(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))
 #Comapre Train and Test scores fort KNN
print('Train Score: ',model.score(X_train, Y_train))
print('Test Score: ',model.score(X_test,Y_test))


 
 #Split data into train and test sets
Z_train,Z_test,Y_train,Y_test = train_test_split(Z,Y,test_size=0.3)
 
 #Predict Output
Y_pred = model.predict(Z_test)
print('KNN Accuracy: ',metrics.accuracy_score(Y_test,Y_pred))
print(confusion_matrix(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))

 #Comapre Train and Test scores fort KNN
print('Test Score: ',model.score(Z_test,Y_test))

# #================================================================================

# #================================================================================
# 
 #use max, min, and median as the range for the amplitudes.
 #when .describe() is called, their positions are the same
 
 #min
minAmp = data_stats.Heat_Release[3]
 #25%
lower25 = data_stats.Heat_Release[4]
 #50%
medianAmp =data_stats.Heat_Release[5]
 #75%
upper75 = data_stats.Heat_Release[6]
 #max
maxAmp = data_stats.Heat_Release[7]

print('\nHeat Release Min:    ',minAmp)
print('Heat Release 25th  : ',lower25)
print('Heat Release Median: ',medianAmp)
print('Heat Release 75th:   ',upper75)
print('Heat Release Max:    ',maxAmp)
 
 
xMax = 20
plt.plot(Heat_Release_list[:xMax])
#sigPlot.sup_title("Frequency plot of Heat Release")
#sigPlot.set_autoscaley_on(False)
plt.ylim(minAmp,maxAmp)
plt.xlim(0,xMax)
#place horizontal line at y = median
plt.axhline(y=medianAmp, color='r',linewidth=2, linestyle='-')
plt.grid(True)
plt.show()
 
 #compare Heat Release from .58 and .75 datasets
f,(ax1,ax2) = plt.subplots(2, sharex=True,sharey=True)
ax1.plot(Heat_Release_list[:100])
plt.grid(True)
ax1.set_title('Signal Comparison: 0.58 vs 0.75', fontsize=16)
ax2.plot(Test_Data_List[:100],color='r')
f.subplots_adjust(hspace=0)
#plt.setp([ax.get_ticklabels() for ax in f.axes[:-1]],visible=False)
plt.grid(True)
plt.show()
# 
# 
# #===================================================================================


# #===================================================================================

 #wavelet decomposition
import pywt
import matplotlib.pyplot as plt
import numpy as np
 
 #grab entire column
ts = Heat_Release_list[25:45];

(ca, cd) = pywt.dwt(ts,'haar')

cat = pywt.threshold(ca, np.std(ca)/2,mode='soft')
cdt = pywt.threshold(cd, np.std(cd)/2,mode='soft')

#reconstruct signal and store in varible ts_rec for access later
ts_rec = pywt.idwt(cat, cdt, 'haar')

plt.close('all')

plt.subplot(211)
 
 
 # Original coefficients
plt.plot(ca, '--*b')
plt.plot(cd, '--*r')
plt.title('Signal Reconstruction: E058', fontsize=16) 
 # Thresholded coefficients
plt.plot(cat, '--*c')
plt.plot(cdt, '--*m')
plt.ylabel("Amplitudes",fontsize=12)
plt.legend(['ca','cd','ca_thresh', 'cd_thresh'], loc=0)
plt.grid(True)
 
plt.subplot(212)
plt.plot(ts)
plt.plot(ts_rec, 'r')
plt.xlabel("Cycles (T)",fontsize=12)
plt.ylabel("Amplitudes",fontsize=12)
plt.legend(['original signal', 'reconstructed signal'], loc=0)
plt.grid(True)

plt.show()
# =============================================================================

# #===================================================================================


 