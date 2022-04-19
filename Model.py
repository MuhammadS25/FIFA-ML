import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

# pd.set_option("display.max_rows",None, "display.max_columns", None)
# Load players data
data = pd.read_csv('player-value-prediction.csv')
# checking if data read correctly
# print(data.head())
# print(data.isna().sum())
# Pre-processing
##Droping near empty columns
droped_columns = []
maxMissingValues = 7000
for column in data:
    if data[column].isna().sum() > maxMissingValues:
        droped_columns.append(column)
data.drop(columns=droped_columns, inplace=True, axis=1)

##Droping rows with missing values
data.dropna(inplace=True, axis=0)

##splitting position column into separate columns
newColumns = ['w', 'x', 'y', 'z']
data[newColumns] = data['positions'].str.split(',', expand=True)

##Replacing 'None' entries with 0's
for col in newColumns:
    data[col].fillna(0, inplace=True)
    ##Replacing nominal entries with 1's
    data.loc[data[col] != 0, col] = 1

##Merging the three tables values into position column and droping temporary columns
data['positions'] = data['w'] + data['x'] + data['y'] + data['z']
data['positions'] = data['positions'].astype(np.int64)
data.drop(columns=newColumns, inplace=True, axis=1)

##splitting work_rate column to Attacking_Work_Rate and Defensive Work Rate
data[['Attacking Work Rate', 'Defensive Work Rate']] = data['work_rate'].str.split('/', expand=True)
data.drop(columns=['work_rate'], inplace=True, axis=1)
data['Attacking Work Rate'] = data['Attacking Work Rate'].replace({"Low": 1, "Medium": 2, "High": 3})
data['Defensive Work Rate'] = data['Defensive Work Rate'].replace({" Low": 1, " Medium": 2, " High": 3})
valueCol = data['value']
data.drop(columns=['value'], inplace=True, axis=1)
data.insert(loc=len(data.columns), column='value', value=valueCol)

##bodyType
'''lbl = LabelEncoder()
lbl.fit(list(data['body_type'].values))
data['body_type'] = lbl.transform(list(data['body_type'].values))
lbl.fit(list(data['club_position'].values))
data['club_position'] = lbl.transform(list(data['club_position'].values))
'''
data['body_type'] = data['body_type'].replace(
    {"Stocky": 1, "Normal": 2, "Lean": 3, "Akinfenwa": 1, "Neymar": 3, "C. Ronaldo": 3, "PLAYER_BODY_TYPE_25": 3})

##Encoding Nationality , club_team , preferred_foot , club_position columns
lbl = LabelEncoder()
lbl.fit(list(data['nationality'].values))
data['nationality'] = lbl.transform(list(data['nationality'].values))
lbl.fit(list(data['club_team'].values))
data['club_team'] = lbl.transform(list(data['club_team'].values))
lbl.fit(list(data['preferred_foot'].values))
data['preferred_foot'] = lbl.transform(list(data['preferred_foot'].values))
lbl.fit(list(data['club_position'].values))
data['club_position'] = lbl.transform(list(data['club_position'].values))

##Handling +2 in columns
index_LS = data.columns.get_loc("LS")
index_RB = data.columns.get_loc("RB")
temp = data.iloc[:, index_LS:index_RB + 1]
for col in temp:
    temp[col] = temp[col].str[0:2]
    temp[col] = temp[col].astype(np.int64)

data.iloc[:, index_LS:index_RB + 1] = temp

##using last 2 digits of year in club_join_date & contract_end_year
data['club_join_date'] = data['club_join_date'].str[-2:]
data['club_join_date'] = data['club_join_date'].astype(np.int64)
data['contract_end_year'] = data['contract_end_year'].str[-2:]
data['contract_end_year'] = data['contract_end_year'].astype(np.int64)

# data.drop(columns=['traits'],inplace=True,axis=1) traits aslun msh mwgooda dropped!!

##
# Get the correlation between the features
corr = data.corr()
# Top 50% Correlation training features with the Value
top_feature = corr.index[abs(corr['value']) > 0.0]
top_corr = data[top_feature].corr()
top_feature = top_feature.delete(-1)

# Features
X = data[top_feature]
# print(top_feature)
# print("coro equals ")
# print(top_corr)
# print("data equals ")
# print(X)
# X=data.iloc[:,4:-1]
# print(X['positions'])
# print(X.info())
# label
Y = data['value']
# print(Y)

# Split the data to training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=True, random_state=10)

# X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size = 0.50,shuffle=True)

poly_features = PolynomialFeatures(degree=3)

# transforms the existing features to higher degree features.
X_train_poly = poly_features.fit_transform(X_train)

# fit the transformed features to Linear Regression
poly_model = linear_model.LinearRegression()
poly_model.fit(X_train_poly, y_train)

# predicting on training data-set
y_train_predicted = poly_model.predict(X_train_poly)
ypred = poly_model.predict(poly_features.transform(X_test))

# predicting on test data-set
prediction = poly_model.predict(poly_features.fit_transform(X_test))

print('Co-efficient of linear regression', poly_model.coef_)
print('Intercept of linear regression model', poly_model.intercept_)
print('Mean Square Error', metrics.mean_squared_error(y_test, prediction))

true_player_value = np.asarray(y_test)[0]
predicted_player_value = prediction[0]
print('True value for the first player in the test set in millions is : ' + str(true_player_value))
print('Predicted value for the first player in the test set in millions is : ' + str(predicted_player_value))
