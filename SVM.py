import time

from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
import joblib
def SVM_Model(X_train, X_test, y_train, y_test):
    start = time.time()
    svm_kernel_ovr = OneVsRestClassifier(SVC(kernel='poly', degree=3)).fit(X_train, y_train)
    end = time.time()
    testStart = time.time()
    accuracy = svm_kernel_ovr.score(X_test, y_test)
    testEnd = time.time()
    print("Accuracy for SVM Model --> ", accuracy)
    print(f"Training time is: {end - start}")
    print(f"Testing time is: {testEnd - testStart}")
    print("---------------------------------------------")
    # SVM_Model.Accuracy = accuracy
    # SVM_Model.trainTime = end - start
    # SVM_Model.testTime = testEnd - testStart