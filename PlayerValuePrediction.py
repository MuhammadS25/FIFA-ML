import pandas as pd
from sklearn.model_selection import train_test_split
from Preprocessing import *
from Polynomial_Regression import *
from MultiVariable_Regression import *

# Load players data
data = pd.read_csv('player-value-prediction.csv')

# PreProcessing
data = pre_processing(data)

# Get the correlation between the features
# Correlation Plotting

data, top_features = Correlation_Plotting(data)

# Features
X = data[top_features]
# Label
Y = data['value']

# Split the data to training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=True, random_state=105)

# Polynomial Regression Model
PolynomialRegression(X_train, X_test, y_train, y_test, X, Y)

# MultiVariable Regression Model
MultiVariableRegression(X_train, X_test, y_train, y_test)
