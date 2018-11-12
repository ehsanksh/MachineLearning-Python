import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
#Assumed you have, X (attributes) for training data set and x_test(attributes) of test_dataset
# Create KNeighbors classifier object model 
model = KMeans(n_clusters=3, random_state=0)
# Train the model using the training sets and check score
model.fit(X)
#Predict Output
predicted= model.predict(test_x)
#######################################
from sklearn import cluster, datasets
iris = datasets.load_iris()
k_means = cluster.KMeans()
k_means.fit(iris.data)
k_means.labels_[::10]
iris.target[::10]