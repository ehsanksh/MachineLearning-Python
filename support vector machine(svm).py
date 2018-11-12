import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create SVM classification object 
model = svm.SVC() # there is various option associated with it, this is simple for classification. You can refer link, for mo# re detail.
# Train the model using the training sets and check score
model.fit(X, y)
model.score(X, y)
#Predict Output
predicted= model.predict(test_X)
#################################################

from sklearn import svm
clf = svm.LinearSVC()
clf.fit(iris.data, iris.target) # learn from the data 
clf.predict([[ 5.0,  3.6,  1.3,  0.25]])
clf.coef_
################################################
from sklearn import svm
svc = svm.SVC(kernel='linear')
svc.fit(iris.data, iris.target) 
##############
#kernels
svc = svm.SVC(kernel='linear')
svc = svm.SVC(kernel='rbf')