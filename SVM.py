from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
import joblib
def SVM_Model(X_train, X_test, y_train, y_test):
    svm_kernel_ovr = OneVsRestClassifier(SVC(kernel='poly', degree=3)).fit(X_train, y_train)
    joblib.dump(svm_kernel_ovr,"SVM.sav")
    accuracy = svm_kernel_ovr.score(X_test, y_test)
    print("Accuracy for SVM Model --> ", accuracy)

