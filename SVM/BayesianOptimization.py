import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

# Load your features and labels
X, y = load_data()  # Replace with your function to load features and labels

# Create a SVM classifier
svm_clf = SVC()

# Specify hyperparameters and their ranges
param_space = {
    "C": Real(1e-3, 1e3, prior="log-uniform"),
    "kernel": Categorical(["linear", "rbf"]),
    "gamma": Categorical(["scale", "auto"]) + Real(1e-3, 1e3, prior="log-uniform"),
}

# Perform Bayesian optimization
bayes_search = BayesSearchCV(
    svm_clf, search_spaces=param_space, n_iter=100, cv=5, n_jobs=-1, verbose=2
)
bayes_search.fit(X, y)

# Print the best hyperparameters and their corresponding score
print(f"Best hyperparameters: {bayes_search.best_params_}")
print(f"Best score: {bayes_search.best_score_}")

# Evaluate the model
y_true, y_pred = y_test, bayes_search.predict(X_test)
print(classification_report(y_true, y_pred))
