import time

import joblib
from sklearn.tree import DecisionTreeClassifier
from Preprocessing import *

def DecisionTree_Model(X_train, X_test, y_train, y_test):
    clf = DecisionTreeClassifier(max_depth=10)
    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    testStart = time.time()
    prediction = clf.predict(X_test)
    testEnd = time.time()
    accuracy = np.mean(prediction == y_test)
    print("Accuracy For DecisionTree Model --> ", accuracy)
    print(f"Training time is: {end - start}")
    print(f"Testing time is: {testEnd - testStart}")
    # DecisionTree_Model.Accuracy = accuracy
    # DecisionTree_Model.trainTime = end - start
    # DecisionTree_Model.testTime = testEnd - testStart