
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
X, y = np.arange(10).reshape((5, 2)), range(5)
X
list(y)

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.33, random_state=41)

########################################################

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load dataset.
iris = load_iris()
X, y = iris.data, iris.target

iris.data.shape
iris.target.shape

np.unique(iris.target)
# split data into training and test data.
train_X, test_X, train_y, test_y = train_test_split(X, y, 
                                                    train_size=0.5,
                                                    test_size=0.5,
                                                    random_state=123)
print("Labels for training and testing data")
print(train_y)
print(test_y)
######################################################
from sklearn.datasets import load_digits
digits = load_digits()
digits.data.shape
digits.target.shape

digits.images.shape

######################################################
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()
diabetes.data.shape
diabetes.target.shape


