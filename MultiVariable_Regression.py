import time
from sklearn import linear_model
from sklearn import metrics


def MultiVariableRegression(X_train, X_test, y_train, y_test):
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
