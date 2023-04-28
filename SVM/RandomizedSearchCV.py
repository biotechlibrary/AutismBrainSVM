import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Load your features and labels
X, y = load_data()  # Replace with your function to load features and labels

# Create a SVM classifier
svm_clf = SVC()

# Specify hyperparameters and their ranges
param_dist = {
    "C": np.logspace(-3, 3, 7),
    "kernel": ["linear", "rbf"],
    "gamma": ["scale", "auto"] + list(np.logspace(-3, 3, 7)),
}

# Perform randomized search
random_search = RandomizedSearchCV(
    svm_clf, param_distributions=param_dist, n_iter=100, cv=5, n_jobs=-1, verbose=2
)
random_search.fit(X, y)

# Print the best hyperparameters and their corresponding score
print(f"Best hyperparameters: {random_search.best_params_}")
print(f"Best score: {random_search.best_score_}")

# Evaluate the model
y_true, y_pred = y_test, random_search.predict(X_test)
print(classification_report(y_true, y_pred))
