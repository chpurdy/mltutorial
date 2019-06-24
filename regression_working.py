import pandas as pd
import numpy as np 
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

# data from https://archive.ics.uci.edu/ml/datasets/Student+Performance

data = pd.read_csv("student-mat.csv", sep=';')

data = data[["G1","G2","G3","studytime","failures","absences"]]

predict = "G3" # this is the attribute we are going to try to predict

X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])

x_train, y_train, x_test, y_test = sklearn.model_selection.train_test_split(X,Y, test_size=0.1)
