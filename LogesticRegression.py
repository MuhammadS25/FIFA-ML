from sklearn.model_selection import train_test_split
from Preprocessing import *
from sklearn.linear_model import LogisticRegression


def LogisticRegressionModel(X_train, X_test, y_train, y_test):
    model = LogisticRegression(multi_class='multinomial', solver='saga')
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    print(np.mean(pred == y_test))
