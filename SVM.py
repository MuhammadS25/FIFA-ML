from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

def SVM_Model(X_train, X_test, y_train, y_test):
    svm_kernel_ovr = OneVsRestClassifier(SVC(kernel='poly', degree=3)).fit(X_train, y_train)
    accuracy = svm_kernel_ovr.score(X_test, y_test)
    print("Accuracy for SVM Model --> ", accuracy)

