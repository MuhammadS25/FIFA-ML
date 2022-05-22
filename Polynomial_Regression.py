import numpy as np
import time
from sklearn import linear_model
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt



def PolynomialRegression(X_train, X_test, y_train, y_test, X, Y):
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

    true_player_value = np.asarray(y_test)[0]
    predicted_player_value = prediction[0]
    print('True value for the first player in the test set is ' + str(true_player_value))
    print('Predicted value for the first player in the test set is ' + str(predicted_player_value))
    print('===========================================================================')
