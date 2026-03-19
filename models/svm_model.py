from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV


def train_model(X_train, y_train):

    # Define parameter grid for tuning
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'kernel': ['rbf', 'linear']
    }

    # Use GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(
        SVC(probability=True),  # Enable probability for predict_proba
        param_grid,
        cv=3,  # 3-fold cross-validation
        scoring='accuracy',
        n_jobs=-1  # Use all available cores
    )

    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    print("Classification Report")

    print(classification_report(y_test, y_pred))

    print("Confusion Matrix")

    print(confusion_matrix(y_test, y_pred))

    return y_pred