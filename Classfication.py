from DecisionTree import DecisionTree_Model
from LogisticRegression import *
import time

import matplotlib.pyplot as plt

# Load players data
from SVM import SVM_Model

data = pd.read_csv('player-classification.csv')
# PreProcessing
data = pre_processing(data, 'PlayerLevel')
data['PlayerLevel'] = data['PlayerLevel'].replace({"S": 5, "A": 4, "B": 3,"C":2,"D":1})
# Get the correlation between the features
# Correlation Plotting
data, top_features = Correlation_Plotting(data,'PlayerLevel')

# Features
X = data[top_features]
# Label
Y = data['PlayerLevel']

# Split the data to training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=True, random_state=105)
LogisticRegressionModel(X_train, X_test, y_train, y_test)
SVM_Model(X_train, X_test, y_train, y_test)
DecisionTree_Model(X_train, X_test, y_train, y_test)





