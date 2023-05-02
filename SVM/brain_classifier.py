import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

def train_and_evaluate_classifier(features, labels, output_dir):
    """
    Trains and evaluates an SVM classifier using the provided features and labels.

    Args:
        features (numpy.ndarray): The extracted features, with shape (n_subjects, n_components).
        labels (numpy.ndarray): The corresponding labels for the subjects.
        output_dir (str): Path to the output directory to save the analysis results.

    Returns:
        None

    Example:
        >>> train_and_evaluate_classifier(features, labels, '/path/to/output_dir/')
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Define the SVM classifier and hyperparameter search space
    svm = SVC()
    param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf'], 'gamma': [0.1, 1, 10, 100]}

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Train the classifier with the optimal hyperparameters
    best_params = grid_search.best_params_
    best_svm = SVC(**best_params)
    best_svm.fit(X_train, y_train)

    # Evaluate the classifier on the testing set
    y_pred = best_svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))

    # Save the classifier to a file
    output_file = os.path.join(output_dir, 'svm_classifier.pkl')
    joblib.dump(best_svm, output_file)
