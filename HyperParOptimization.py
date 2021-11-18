# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#this is my practice program

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

#import csv file
data = pd.read_csv("diabetes.csv")

#view first 20 rows of data
print(data.head(20))

#view stats about all dataset
print(data.describe())

#some rows have missing values or "zeros"/ to count we must reimport without column names
data = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",header=None)
# print the sum of the number of zeros in the first 5 rows
print((data[[1,2,3,4,5]]==0).sum())

#mark zeros as missing with 'NaN'
data[[1,2,3,4,5]] = data[[1,2,3,4,5]].replace(0, np.NaN)

#show number of times 
print(data.isnull(sum()))


#so we need to replace zero with the averages of that column
 



