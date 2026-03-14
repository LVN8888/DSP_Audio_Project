from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def train_model(X_train, y_train):

    model = SVC(
        kernel="rbf",
        C=10,
        gamma="scale"
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    print("Classification Report")

    print(classification_report(y_test, y_pred))

    print("Confusion Matrix")

    print(confusion_matrix(y_test, y_pred))

    return y_pred