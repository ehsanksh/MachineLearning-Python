import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
X, y = np.arange(14).reshape((7,2)), np.array([0.1, 0.23, 0.31, 0.41, 0.76, 0.86, 1])


# split data into training and test data.
train_x, test_x, train_y, test_y = train_test_split(X, y, 
                                                    train_size=0.5,
                                                    test_size=0.5,
                                                    random_state=42)
from sklearn import decomposition
#Assumed you have training and test data set as train and test
# Create PCA obeject 
k =min(n_sample, n_features)
pca= decomposition.PCA(n_components=k) #default value of k =min(n_sample, n_features)
# For Factor analysis
#fa= decomposition.FactorAnalysis()
# Reduced the dimension of training dataset using PCA
train_reduced = pca.fit_transform(train)
#Reduced the dimension of test dataset
test_reduced = pca.transform(test)
######################################################
from sklearn import decomposition
pca = decomposition.PCA(n_components=2)
pca.fit(iris.data)
X = pca.transform(iris.data)
plt.scatter(X[:, 0], X[:, 1], c=iris.target) 
