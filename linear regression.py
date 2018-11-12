# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 11:44:50 2018

@author: lenovo
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy 
import pandas
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Load dataset.
X, y = np.arange(14).reshape((7,2)), np.array([0.1, 0.23, 0.31, 0.41, 0.76, 0.86, 1])


# split data into training and test data.
train_x, test_x, train_y, test_y = train_test_split(X, y, 
                                                    train_size=0.5,
                                                    test_size=0.5,
                                                    random_state=42)

#Load Train and Test datasets
#Identify feature and response variable(s) and values must be numeric and numpy arrays

# Create linear regression object
linear = linear_model.LinearRegression()
# Train the model using the training sets and check score
linear.fit(train_x, train_y)
linear.score(train_x, train_y)
#Equation coefficient and Intercept
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)
#Predict Output
predicted= linear.predict(test_x)