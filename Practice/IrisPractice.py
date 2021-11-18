#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 08:21:58 2019

@author: 1vn
"""

#This program is based off of the Iris Dataset
#from sklearn import datasets, svm

#library of various plots
import matplotlib.pyplot as plot

# Load libraries
import pandas
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
gamma='auto'
#solver = 'randomized'



##datasets is a dictionary like object that holds all data and metadata
#iris = datasets.load_iris()
#digits =datasets.load_digits()
#
##.data and .target are members of the data set
##data gives us the features  used rto classify
#print("Data",digits.data)
#
##target give us the number corresponding to each picture
#print("Target",digits.target)
#
##clf is a classifier estimator that will learn
#clf = svm.SVC(gamma=0.001, C=100.)
#
##data must be fitted to the model first---> fit also means learn from 
#clf.fit(digits.data[:-1], digits.target[:-1])  
##now we can predict
#clf.predict(digits.data[:-1])
#
#plot.figure(1, figsize=(3,3))
#plot.imshow(digits.images[-1], cmap=plot.cm.gray_r, interpolation='nearest')
##prints graphic representation of image
#plot.show()

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

#print("shape",dataset.shape)
#print("head",dataset.head(20))
#print("tail",dataset.tail(20))
#
#print("describe", dataset.describe())
#print("size or # of instances", dataset.groupby('class').size())

#univariate(single varible) plots are good for understanding each attribute
dataset.plot(kind='box', subplots=False, layout=(2,2), sharex=False, sharey=False)
plot.show()

dataset.hist()

#multivariate plots ---> relationship between attributes
scatter_matrix(dataset)
plot.show

#Split dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

#Test options and evaluation metric
seed = 7
scoring = 'accuracy'


#build a model to represent data
# FIRST LETS Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

#Then loop through each models representaion of data
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

#Choose the best model
fig = plot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plot.boxplot(results)
ax.set_xticklabels(names)
plot.show()

#make predictions
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))



