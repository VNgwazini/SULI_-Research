#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 14:55:45 2019

@author: 1vn
"""

#import datasets and library and create object
import sklearn
from sklearn.decomposition.pca import PCA
from sklearn import datasets
from sklearn.preprocessing import scale
from sklearn.model_selection._split import train_test_split
from sklearn import cluster
# Import `Isomap()`
from sklearn.manifold import Isomap

import numpy as np
#import matplotlib for visualization like matlab
import matplotlib.pyplot as plot




#load data into variable from library
myData = datasets.load_digits()

#print("-------------DIGIT DATA--------------")
#contains the raw data and the attributes of that data
print('Digit Data: \n',myData)
#print("-------------DIGIT DATA--------------")


#dataset keys
#print("-------------DIGIT KEYS--------------")
#contains the attributes of the data that can be accessed using .data ect...
print('Digit Keys: \n',myData.keys())
#print("-------------DIGIT KEYS--------------")


#print("-------------DATA SHAPE--------------")
myData_data = myData.data
print('Data Shape: ',myData_data.shape)
#print("-------------DATA SHAPE--------------")


#print("-------------IMAGE RESHAPE MATCH ??--------------")
print('data reshaped: \n',np.all(myData.images.reshape((1797,64)) == myData.data))
#print("-------------IMAGE RESHAPE MATCH ??--------------")




#plot the points on the graph
figure = plot.figure(figsize=(6,6))
figure.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.5, wspace=0.5)

for i in range(64):
    axis =figure.add_subplot(8,8, i + 1, xticks=[], yticks=[])
    #show image of each plot
    axis.imshow(myData.images[i], cmap=plot.cm.binary, interpolation='nearest')
    axis.text(0,7,str(myData.target[i]))

#print("-----------------------------------------------PLOT------------------------------------------------")
plot.show()
#print("-----------------------------------------------PLOT------------------------------------------------")




#JOIN IMAGES AND TARGET LABELS
images_and_labels = list(zip(myData.images,myData.target))

for index, (image, label) in enumerate(images_and_labels[:8]):
    plot.subplot(2,4,index + 1)
    plot.axis('off')
    #show image of each plot
    plot.imshow(image,cmap=plot.cm.gray_r,interpolation='nearest')
    plot.title('Training: '+ str(label))

#print("-------------------------------------------------LABELED PLOT-----------------------------------------\n")
plot.show()
#print("-----------------------------------------------LABELED PLOT------------------------------------------------\n")






#create random Principal Component Analysis Model to comapre data dimensions to # of features
randomized_pca = PCA(svd_solver='randomized',n_components=2)
#same but regular/notrandom pca
pca = PCA(n_components=2)

#fit data to that model
reduced_data_rpca = pca.fit_transform(myData.data,y=None)
#same but for regular pca
reduced_data_pca = pca.fit_transform(myData.data,y=None)

##get the shape and analyze it
reduced_data_rpca.shape
reduced_data_pca.shape

#print(reduced_data_rpca)
#print(reduced_data_pca)

colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']
for i in range(len(colors)):
    x = reduced_data_rpca[:, 0][myData.target == i]
    y = reduced_data_rpca[:, 1][myData.target == i]
    plot.scatter(x, y, c=colors[i])
plot.legend(myData.target_names, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plot.xlabel('First Principal Component')
plot.ylabel('Second Principal Component')
plot.title("PCA Scatter Plot")
plot.show()





#scale data -> shift distribtuion of attributes so that mean is 0 and std is 1
data = scale(myData.data)

#split "myData" dataset into train and test sample sets
X_train, X_test, Y_train, Y_test, images_train, images_test =train_test_split(data, myData.target, myData.images, test_size=0.25, random_state=42)

#set # of samples and features based on X training
n_samples,n_features,= X_train.shape

#set # of training labels based on Y training
n_myData =len(np.unique(Y_train))

#inspect X
print('n_samples: ',n_samples)
print('n_feature: ',n_features)

#inspect Y training
print("y Training: ",len(Y_train))

#create KMeans model
clf = cluster.KMeans(init='k-means++', n_clusters=10, random_state=42)

#fit training data to model
clf.fit(X_train)

# Figure size in inches
fig = plot.figure(figsize=(8, 3))

# Add title
fig.suptitle('Cluster Center Images', fontsize=14, fontweight='bold')

# For all labels (0-9)
for i in range(10):
    # Initialize subplots in a grid of 2X5, at i+1th position
    ax = fig.add_subplot(2, 5, 1 + i)
    # Display images
    ax.imshow(clf.cluster_centers_[i].reshape((8, 8)), cmap=plot.cm.binary)
    # Don't show the axes
    plot.axis('off')

# Show the plot
plot.legend(myData.target_names, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plot.show()

# Predict the labels for `X_test`
Y_pred=clf.predict(X_test)

# Print out the first 100 instances of `y_pred`
print('Y Prediction\n',Y_pred[:100])

# Print out the first 100 instances of `y_test`
print('Y Test\n',Y_test[:100])

# Study the shape of the cluster centers
clf.cluster_centers_.shape 
#---------------------------------------------------------------------------
# Create an isomap and fit the `digits` data to it
X_iso = Isomap(n_neighbors=10).fit_transform(X_train)

# Compute cluster centers and predict cluster index for each sample
clusters = clf.fit_predict(X_train)

# Create a plot with subplots in a grid of 1X2
fig, ax = plot.subplots(1, 2, figsize=(8, 4))

# Adjust layout
fig.suptitle('Predicted Versus Training Labels', fontsize=14, fontweight='bold')
fig.subplots_adjust(top=0.85)

# Add scatterplots to the subplots 
ax[0].scatter(X_iso[:, 0], X_iso[:, 1], c=clusters)
ax[0].set_title('Predicted Training Labels')
ax[0].legend(myData.target_names, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax[1].scatter(X_iso[:, 0], X_iso[:, 1], c=Y_train)
ax[1].set_title('Actual Training Labels')
ax[1].legend(myData.target_names, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

# Show the plots
plot.show()
#--------------------------------------------------------------------------
# Import `PCA()`
from sklearn.decomposition import PCA

# Model and fit the `digits` data to the PCA model
X_pca = PCA(n_components=2).fit_transform(X_train)

# Compute cluster centers and predict cluster index for each sample
clusters = clf.fit_predict(X_train)

# Create a plot with subplots in a grid of 1X2
fig, ax = plot.subplots(1, 2, figsize=(8, 4))

# Adjust layout
fig.suptitle('Predicted Versus Training Labels', fontsize=14, fontweight='bold')
fig.subplots_adjust(top=0.85)

# Add scatterplots to the subplots 
ax[0].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters)
ax[0].set_title('Predicted Training Labels')
ax[1].scatter(X_pca[:, 0], X_pca[:, 1], c=Y_train)
ax[1].set_title('Actual Training Labels')

# Show the plots
plot.legend(myData.target_names, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plot.show()

#confirm that our model is best for our data

# Import `metrics` from `sklearn`
from sklearn import metrics

# Print out the confusion matrix with `confusion_matrix()`
print('Confusion Matrix\n',metrics.confusion_matrix(Y_test, Y_pred))

from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, adjusted_mutual_info_score, silhouette_score
print('% 9s' % 'inertia    homo   compl  v-meas     ARI AMI  silhouette')
print('%i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
          %(clf.inertia_,
      homogeneity_score(Y_test, Y_pred),
      completeness_score(Y_test, Y_pred),
      v_measure_score(Y_test, Y_pred),
      adjusted_rand_score(Y_test, Y_pred),
      adjusted_mutual_info_score(Y_test, Y_pred),
      silhouette_score(X_test, Y_pred, metric='euclidean')))
#----------------------------------------------------------------------
# Import `train_test_split`
#from sklearn.cross_validation import train_test_split

# Split the data into training and test sets 
X_train, X_test, Y_train, Y_test, images_train, images_test = train_test_split(myData.data, myData.target, myData.images, test_size=0.25, random_state=42)

# Import the `svm` model
from sklearn import svm

# Create the SVC model 
svc_model = svm.SVC(gamma=0.001, C=100., kernel='linear')


# Split the `digits` data into two equal sets
X_train, X_test, Y_train, Y_test = train_test_split(myData.data, myData.target, test_size=0.5, random_state=0)

# Import GridSearchCV
from sklearn.model_selection._search import GridSearchCV

# Set the parameter candidates
parameter_candidates = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
]


#-----------------------------------------------------------
# Import GridSearchCV
#from sklearn.grid_search import GridSearchCV


# Create a classifier with the parameter candidates
clf2 = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, n_jobs=-1)

# Train the classifier on training data
clf2.fit(X_train, Y_train)

# Print out the results 
print('Best score for training data:', clf2.best_score_)
print('Best `C`:',clf2.best_estimator_.C)
print('Best kernel:',clf2.best_estimator_.kernel)
print('Best `gamma`:',clf2.best_estimator_.gamma)

# Apply the classifier to the test data, and view the accuracy score
svc_model.fit(X_test, Y_test)
clf2.score(X_test, Y_test)  

# Train and score a new classifier with the grid search parameters

print(svm.SVC(C=10, kernel='rbf', gamma=0.001).fit(X_train, Y_train).score(X_test, Y_test))

# Predict the label of `X_test`
print(svc_model.predict(X_test))

# Print `y_test` to check the results
print(Y_test)

# Assign the predicted values to `predicted`
predicted = svc_model.predict(X_test)

# Zip together the `images_test` and `predicted` values in `images_and_predictions`
images_and_predictions = list(zip(images_test, predicted))

# For the first 4 elements in `images_and_predictions`
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    # Initialize subplots in a grid of 1 by 4 at positions i+1
    plot.subplot(1, 4, index + 1)
    # Don't show axes
    plot.axis('off')
    # Display images in all subplots in the grid
    plot.imshow(image, cmap=plot.cm.gray_r, interpolation='nearest')
    # Add a title to the plot
    plot.title('Predicted: ' + str(prediction))

# Show the plot
plot.legend(myData.target_names, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plot.show()

# Import `metrics`
from sklearn import metrics

# Print the classification report of `y_test` and `predicted`
print(metrics.classification_report(Y_test, predicted))

# Print the confusion matrix
print(metrics.confusion_matrix(Y_test, predicted))
#-------------------------------------------------------------------------
# Create an isomap and fit the `digits` data to it
X_iso = Isomap(n_neighbors=10).fit_transform(X_train)

# Compute cluster centers and predict cluster index for each sample
predicted = svc_model.predict(X_train)

# Create a plot with subplots in a grid of 1X2
fig, ax = plot.subplots(1, 2, figsize=(8, 4))

# Adjust the layout
fig.subplots_adjust(top=0.85)

# Add scatterplots to the subplots 
ax[0].scatter(X_iso[:, 0], X_iso[:, 1], c=predicted)
ax[0].set_title('Predicted labels')
ax[1].scatter(X_iso[:, 0], X_iso[:, 1], c=Y_train)
ax[1].set_title('Actual Labels')


# Add title
fig.suptitle('Predicted versus actual labels', fontsize=14, fontweight='bold')

# Show the plot
plot.legend(myData.target_names, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plot.show()



