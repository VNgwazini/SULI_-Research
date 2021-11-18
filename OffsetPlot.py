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

import matplotlib.pyplot as plt


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
col_names=['Heat_Release','DOWN_OR_UP','Cnsctv_N_Y','Bin']


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
         #print(data[eachDF].head(5))
     #    data[eachDF].to_csv(data[eachDF])
     #    data[eachDF].to_csv(EngineData[1][eachDF])
         
    #     print(data[eachDF])
     
         #run some statistics on the data
         data_stats = data[eachDF].describe()
    #     print('Data Description: \n',data_stats)        
         
         #min
#         minAmp = data_stats.Heat_Release[3]
         #25%
         #print('minAmp: ',minAmp)
         lower25 = data_stats.Heat_Release[4]
         #50%
         medianAmp =data_stats.Heat_Release[5]
         #75%
         upper75 = data_stats.Heat_Release[6]
         #max
#         maxAmp = data_stats.Heat_Release[7]
         
         data[eachDF]['DOWN_OR_UP'].at[0] = int(1);
         data[eachDF]['Cnsctv_N_Y'].at[0] = int(0);
         data[eachDF]['Bin'].at[0] = int(1)
         
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

    
         for eachRow in range(0,len(data[0]['Heat_Release'])):
             #=========================Bin============================================
             if(data[eachDF]['Heat_Release'].at[eachRow] <= lower25):
                 #print('\nrow value: ',data[eachDF]['Bin'][eachRow])
                 #print('1')
                 data[eachDF]['Bin'].at[eachRow]= int(1)
                 #print('row value: ',data[eachDF]['Bin'].at[eachRow])
                 
             elif(data[eachDF]['Heat_Release'].at[eachRow] > lower25 and data[eachDF]['Heat_Release'].at[eachRow] <= medianAmp):
                 #print('\nrow value: ',data[eachDF]['Bin'].at[eachRow])
                 #print('2')     
                 data[eachDF]['Bin'].at[eachRow]= int(2)
                 #print('row value: ',data[eachDF]['Bin'].at[eachRow])
                      
             elif(data[eachDF]['Heat_Release'].at[eachRow] > medianAmp and data[eachDF]['Heat_Release'].at[eachRow] <= upper75):
                 #print('\nrow value: ',data[eachDF]['Bin'].at[eachRow])
                 #print('3')     
                 data[eachDF]['Bin'].at[eachRow]= int(3)
                 #print('row value: ',data[eachDF]['Bin'].at[eachRow])
                      
             elif(data[eachDF]['Heat_Release'].at[eachRow] > upper75):
                 #print('\nrow value:    ',data[eachDF]['Bin'].at[eachRow])
                 #print('4')     
                 data[eachDF]['Bin'].at[eachRow]= int(4)
                 #print('row value:    ',data[eachDF]['Bin'].at[eachRow])
             #=========================Bin============================================

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
#        print('\nIn For Loop: \n')
        #    print(data[eachDF])

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
    return Heats, UpDowns, Cnsctvs, Bins


#========================================================================================================================================================================================
#                MAIN FUNCTION
#========================================================================================================================================================================================

def main():
#    print('\nIn main: \n')
    #print(EngineData)
    
    H,U,C,P = CsvToDF(EngineData,fileIndex)
#    print(H[0][0:7])
#    print(H[0][1:8])

    plt.scatter((H[0][0:1007]),(H[0][1:1008]))
    plt.title("E058 Cycle Heat Release: T-1 vs T",fontsize=16)
    plt.xlabel("Cycle T-1",fontsize=12)
    plt.ylabel("Cycle T",fontsize=12)
    plt.show()
    
    plt.scatter((H[1][0:1007]),(H[1][1:1008]))
    plt.title("E060 Cycle Heat Release: T-1 vs T",fontsize=16)
    plt.xlabel("Cycle T-1",fontsize=12)
    plt.ylabel("Cycle T",fontsize=12)
    plt.show()

    plt.scatter((H[2][0:1007]),(H[2][1:1008]))
    plt.title("E065-1 Cycle Heat Release: T-1 vs T",fontsize=16)
    plt.xlabel("Cycle T-1",fontsize=12)
    plt.ylabel("Cycle T",fontsize=12)
    plt.show()
    
    plt.scatter((H[3][0:1007]),(H[3][1:1008]))
    plt.title("E065-2 Cycle Heat Release: T-1 vs T",fontsize=16)
    plt.xlabel("Cycle T-1",fontsize=12)
    plt.ylabel("Cycle T",fontsize=12)
    plt.show()

    plt.scatter((H[4][0:1007]),(H[4][1:1008]))
    plt.title("E075 Cycle Heat Release: T-1 vs T",fontsize=16)
    plt.xlabel("Cycle T-1",fontsize=12)
    plt.ylabel("Cycle T",fontsize=12)
    plt.show()

     #compare all three signals on line graph
    f,(ax1,ax2,ax3,ax4,ax5) = plt.subplots(5,sharex=True,sharey=True)
    ax1.scatter((H[0][0:1007]),(H[0][1:1008]),c='red',label='E058',
               alpha=0.3, edgecolors='none')
    ax1.grid(True)
    ax1.set_title('Signal Comparison: E058 -> E075', fontsize=16)
#-------------------------------------------------------------------------#    
    ax2.scatter((H[1][0:1007]),(H[1][1:1008]),c='orange',label='E060',
               alpha=0.3, edgecolors='none')
    ax2.grid(True)
#-------------------------------------------------------------------------#
    ax3.scatter((H[2][0:1007]),(H[2][1:1008]),c='purple',label='E065-1',
               alpha=0.3, edgecolors='none')
    ax3.grid(True)
#-------------------------------------------------------------------------#    
    ax4.scatter((H[3][0:1007]),(H[3][1:1008]),c='green',label='E065-2',
               alpha=0.3, edgecolors='none')
    ax4.grid(True)
#-------------------------------------------------------------------------#    
    ax5.scatter((H[4][0:1007]),(H[4][1:1008]),c='blue',label='E075',
               alpha=0.3, edgecolors='none')
    ax5.grid(True)
#-------------------------------------------------------------------------#    
    f.subplots_adjust(hspace=0)
    #plt.setp([ax.get_ticklabels() for ax in f.axes[:-1]],visible=False)
    f.legend(loc=5)
    plt.xlabel("Cycle (T)",fontsize=12)
    plt.ylabel('Cycle (T+1)', position=(0.9,0.9))
    plt.grid(True)
    plt.show()


#========================================================================================================================================================================================
#                MAIN FUNCTION
#========================================================================================================================================================================================

    
main()