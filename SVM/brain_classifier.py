import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

def train_and_evaluate_classifier(features, labels, output_dir):
    """
    This function performs the training and evaluation of a Support Vector Machine (SVM) classifier.
    It uses grid search for hyperparameter tuning.

    Parameters
    ----------
    features : np.array
        A numpy array representing the feature matrix.
    labels : np.array
        A numpy array representing the target labels.
    output_dir : str
        The directory where the trained model will be saved.

    Returns
    -------
    None
        The function saves the trained model to a .pkl file and does not return any value.

    """
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    svm = SVC()
    param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': [0.1, 1, 10]}
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_svm = SVC(**best_params)
    best_svm.fit(X_train, y_train)

    y_pred = best_svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))

    output_file = os.path.join(output_dir, 'svm_classifier.pkl')
    joblib.dump(best_svm, output_file)
