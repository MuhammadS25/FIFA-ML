from sklearn import metrics
from sklearn.model_selection import train_test_split
from Preprocessing import *
import joblib

# Load players data
data = pd.read_csv('player-test-samples_pred.csv')

# PreProcessing
data = pre_processing(data, "value")

# Get the correlation between the features
# Correlation PlottingX_train_poly
data, top_features = Correlation_Plotting(data, "value", saved_prediction=True)

# Features
X = data[top_features]
# Label
Y = data['value']

print("Polynomial Regression")
print('----------------------------------------')
polyFeaturesModel = joblib.load("PolynomialFeaturesModel.sav")
Polynomial_Regression = joblib.load("PLinearRegressionModel.sav")

prediction = Polynomial_Regression.predict(polyFeaturesModel.transform(X))
print('R2 =', metrics.r2_score(Y, prediction) * 100)
print('Mean Square Error =', metrics.mean_squared_error(prediction, Y))

true_player_value = np.asarray(Y)[0]
predicted_player_value = prediction[0]
print('True value for the first player in the test set is ' + str(true_player_value))
print('Predicted value for the first player in the test set is ' + str(predicted_player_value))
print('===========================================================================')

print("MultiVariable Regression")
print('----------------------------------------')
MultiVariable_Regression = joblib.load("MultiVariableRegression.sav")

prediction = MultiVariable_Regression.predict(X)
print('R2 =', metrics.r2_score(Y, prediction) * 100)
print('Mean Square Error =', metrics.mean_squared_error(prediction, Y))

true_player_value = np.asarray(Y)[0]
predicted_player_value = prediction[0]
print('True value for the first player in the test set is ' + str(true_player_value))
print('Predicted value for the first player in the test set is ' + str(predicted_player_value))
print('===========================================================================')

data = pd.read_csv('player-test-samples_class.csv')
# PreProcessing
data = pre_processing(data, 'PlayerLevel')
data['PlayerLevel'] = data['PlayerLevel'].replace({"S": 5, "A": 4, "B": 3, "C": 2, "D": 1})
# Get the correlation between the features
# Correlation Plotting
data, top_features = Correlation_Plotting(data, 'PlayerLevel', saved_classification=True)

# Features
X = data[top_features]
# Label
Y = data['PlayerLevel']

print("Logistic Regression")
print('----------------------------------------')
LogisticRegression = joblib.load("LogisticRegressionModel.sav")

prediction = LogisticRegression.predict(X)
accuracy = np.mean(prediction == Y)
print("Accuracy For Logistic Regression Model --> ", accuracy)

true_player_value = np.asarray(Y)[2]
predicted_player_value = prediction[2]
print('True value for the first player in the test set is ' + str(true_player_value))
print('Predicted value for the first player in the test set is ' + str(predicted_player_value))
print('===========================================================================')

print("SVM")
print('----------------------------------------')
SVM = joblib.load("SVM.sav")

prediction = SVM.predict(X)
accuracy = np.mean(prediction == Y)
print("Accuracy For SVM Model --> ", accuracy)

true_player_value = np.asarray(Y)[2]
predicted_player_value = prediction[2]
print('True value for the first player in the test set is ' + str(true_player_value))
print('Predicted value for the first player in the test set is ' + str(predicted_player_value))
print('===========================================================================')

print("Decision Tree")
print('----------------------------------------')
DecisionTree = joblib.load("DecisionTree_Model.sav")

prediction = DecisionTree.predict(X)
accuracy = np.mean(prediction == Y)
print("Accuracy For DecisionTree Model --> ", accuracy)

true_player_value = np.asarray(Y)[2]
predicted_player_value = prediction[2]
print('True value for the first player in the test set is ' + str(true_player_value))
print('Predicted value for the first player in the test set is ' + str(predicted_player_value))
print('===========================================================================')
