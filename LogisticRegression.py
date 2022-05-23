import time

import joblib
from sklearn.model_selection import train_test_split
from Preprocessing import *
from sklearn.linear_model import LogisticRegression


def LogisticRegressionModel(X_train, X_test, y_train, y_test):
    model = LogisticRegression(multi_class='multinomial', solver='saga')
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    testStart = time.time()
    prediction = model.predict(X_test)
    testEnd = time.time()
    accuracy = np.mean(prediction == y_test)
    print("Accuracy For LogisticRegression Model --> ", accuracy)
    print(f"Training time is: {end - start}")
    print(f"Testing time is: {testEnd - testStart}")
    print("---------------------------------------------")
    # LogisticRegressionModel.Accuracy = accuracy
    # LogisticRegressionModel.trainTime = end - start
    # LogisticRegressionModel.testTime = testEnd - testStart