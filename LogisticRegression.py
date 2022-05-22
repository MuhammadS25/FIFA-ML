import joblib
from sklearn.model_selection import train_test_split
from Preprocessing import *
from sklearn.linear_model import LogisticRegression


def LogisticRegressionModel(X_train, X_test, y_train, y_test):
    model = LogisticRegression(multi_class='multinomial', solver='saga')
    model.fit(X_train, y_train)
    joblib.dump(model,"LogisticRegressionModel.sav")
    prediction = model.predict(X_test)
    accuracy = np.mean(prediction == y_test)
    print("Accuracy For LogisticRegression Model --> ", accuracy)
