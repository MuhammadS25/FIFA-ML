import joblib
from sklearn.tree import DecisionTreeClassifier
from Preprocessing import *

def DecisionTree_Model(X_train, X_test, y_train, y_test):
    clf = DecisionTreeClassifier(max_depth=10)
    clf.fit(X_train, y_train)
    joblib.dump(clf,"DecisionTree_Model.sav")
    prediction = clf.predict(X_test)
    accuracy = np.mean(prediction == y_test)
    print("Accuracy For DecisionTree Model --> ", accuracy)
