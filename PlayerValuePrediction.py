from sklearn import linear_model
from sklearn import metrics
import time
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns
from Preprocessing import *

# Load players data
data = pd.read_csv('player-value-prediction.csv')

# PreProcessing
data = pre_processing(data)

# Get the correlation between the features
corr = data.corr()
# Top 50% Correlation training features with the Value
top_features = corr.index[abs(corr['value']) > 0.5]
top_features = top_features.delete(-1)

# Correlation Plotting
plt.subplots(figsize=(12, 8))
top_corr = data[top_features].corr()
sns.heatmap(top_corr, annot=True)
plt.show()

# scaling all the features between 0 and 1 values --> [Normalization]
min_max_scaler = preprocessing.MinMaxScaler()
data[top_features] = min_max_scaler.fit_transform(data[top_features])

# Features
X = data[top_features]
# Label
Y = data['value']

# Split the data to training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=True, random_state=105)

poly_features = PolynomialFeatures(degree=3)

# transforms the existing features to higher degree features.
X_train_poly = poly_features.fit_transform(X_train)

# fit the transformed features to Linear Regression
poly_model = linear_model.LinearRegression()
start = time.time()
poly_model.fit(X_train_poly, y_train)
end = time.time()
print("Polynomial Regression")
print('----------------------------------------')
print(f"Training time is: {end - start}")

# predicting on test data-set
prediction = poly_model.predict(poly_features.fit_transform(X_test))

# Line Plotting
for plot in X.columns:
    plt.scatter(X[plot], Y)
plt.xlabel('PlayerData', fontsize=20)
plt.ylabel('Value', fontsize=20)
plt.show()
print('R2 =', metrics.r2_score(y_test, prediction) * 100)
print('Mean Square Error =', metrics.mean_squared_error(y_test, prediction))

true_player_value = np.asarray(y_test)[50]
predicted_player_value = prediction[50]
print('True value for the first player in the test set in millions is = ' + str(true_player_value))
print('Predicted value for the first player in the test set in millions is = ' + str(predicted_player_value))
print('===========================================================================')
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
