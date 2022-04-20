import pandas as pd
from sklearn.model_selection import train_test_split
from Preprocessing import *
from Polynomial_Regression import *

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

#MultiVariable Regression Model
model2 = linear_model.LinearRegression()
start = time.time()
model2.fit(X_train, y_train)
end = time.time()
print("MultiVariable Regression")
print('----------------------------------------')
print(f"Training time is: {end - start}")

# predicting on training data-set
prediction2 = model2.predict(X_test)

print('Mean Square Error =', metrics.mean_squared_error(y_test, prediction2))
print('R2 =', metrics.r2_score(y_test, prediction2) * 100)
