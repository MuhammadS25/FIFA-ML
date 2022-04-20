import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import metrics
import time
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns

# Load players data
data = pd.read_csv('player-value-prediction.csv')
# checking if data read correctly
# Pre-processing
##Droping near empty columns
droped_columns = []
maxMissingValues = 7000
for column in data:
    if data[column].isna().sum() > maxMissingValues:
        droped_columns.append(column)
data.drop(columns=droped_columns, inplace=True, axis=1)

# Droping rows with missing values
data.dropna(inplace=True, axis=0)

# splitting position column into separate columns
newColumns = ['w', 'x', 'y', 'z']
data[newColumns] = data['positions'].str.split(',', expand=True)

# Replacing 'None' entries with 0's
for col in newColumns:
    data[col].fillna(0, inplace=True)
    ##Replacing nominal entries with 1's
    data.loc[data[col] != 0, col] = 1

# Merging the three tables values into position column and droping temporary columns
data['positions'] = data['w'] + data['x'] + data['y'] + data['z']
data['positions'] = data['positions'].astype(np.int64)
data.drop(columns=newColumns, inplace=True, axis=1)

# splitting work_rate column to Attacking_Work_Rate and Defensive Work Rate
data[['Attacking Work Rate', 'Defensive Work Rate']] = data['work_rate'].str.split('/', expand=True)
data.drop(columns=['work_rate'], inplace=True, axis=1)
data['Attacking Work Rate'] = data['Attacking Work Rate'].replace({"Low": 1, "Medium": 2, "High": 3})
data['Defensive Work Rate'] = data['Defensive Work Rate'].replace({" Low": 1, " Medium": 2, " High": 3})
valueCol = data['value']
data.drop(columns=['value'], inplace=True, axis=1)
data.insert(loc=len(data.columns), column='value', value=valueCol)

# bodyType

data['body_type'] = data['body_type'].replace(
    {"Stocky": 1, "Normal": 2, "Lean": 3, "Akinfenwa": 1, "Neymar": 3, "C. Ronaldo": 3, "PLAYER_BODY_TYPE_25": 3})

# Encoding Nationality , club_team , preferred_foot , club_position columns
lbl = LabelEncoder()
lbl.fit_transform(list(data['nationality'].values))
lbl.fit_transform(list(data['club_team'].values))
lbl.fit_transform(list(data['preferred_foot'].values))
lbl.fit_transform(list(data['club_position'].values))

# Handling +2 in columns
index_LS = data.columns.get_loc("LS")
index_RB = data.columns.get_loc("RB")
temp = data.iloc[:, index_LS:index_RB + 1]

for col in temp:
    temp[col] = temp[col].str[0:2]
    temp[col] = temp[col].astype(np.int64)

data.iloc[:, index_LS:index_RB + 1] = temp

# using last 2 digits of year in club_join_date & contract_end_year
data['club_join_date'] = data['club_join_date'].str[-2:]
data['club_join_date'] = data['club_join_date'].astype(np.int64)
data['contract_end_year'] = data['contract_end_year'].str[-2:]
data['contract_end_year'] = data['contract_end_year'].astype(np.int64)

# data.drop(columns=['traits'],inplace=True,axis=1) traits aslun msh mwgooda dropped!!

# Get the correlation between the features
corr = data.corr()
# Top 50% Correlation training features with the Value
top_features = corr.index[abs(corr['value']) > 0.5]
top_features = top_features.delete(-1)
print(top_features)

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
print(f"Training time of polynomial Regression is: {end - start}")

# predicting on test data-set
prediction = poly_model.predict(poly_features.fit_transform(X_test))

# Line Plotting
for plot in X.columns:
    plt.scatter(X[plot], Y)
plt.xlabel('PlayerData', fontsize=20)
plt.ylabel('Value', fontsize=20)
plt.show()

print('Mean Square Error for Polynomial Regression =', metrics.mean_squared_error(y_test, prediction))

true_player_value = np.asarray(y_test)[0]
predicted_player_value = prediction[0]
print('True value for the first player in the test set in millions is = ' + str(true_player_value))
print('Predicted value for the first player in the test set in millions is = ' + str(predicted_player_value))

model2 = linear_model.LinearRegression()
start = time.time()
model2.fit(X_train, y_train)
end = time.time()
print(f"Training time of MultiVariable Regression is: {end - start}")

# predicting on training data-set
prediction2 = model2.predict(X_test)

print('Mean Square Error for MultiVariable Regression =', metrics.mean_squared_error(y_test, prediction2))
#done