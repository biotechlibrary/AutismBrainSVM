"""
brain_classifier.py

This script trains an SVM classifier on fMRI data to differentiate between autistic and control subjects.
It uses the Yeo 2011 atlas for feature extraction and evaluates the classifier on a test set.

Usage:
    python brain_classifier.py

Requirements:
    - numpy
    - pandas
    - nilearn
    - scikit-learn
    - cuml
    - tqdm
"""

import os
import numpy as np
import pandas as pd
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
from nilearn import image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from cuml import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
import random
import glob
from concurrent.futures import ThreadPoolExecutor
import concurrent

random.seed(42)
np.random.seed(42)

# Load the atlas
atlas = datasets.fetch_atlas_yeo_2011()
masker = NiftiLabelsMasker(labels_img=atlas.thick_7, standardize=False)

# Rest of the script


# Prepare the data
data_path = '/home/pau/micromamba/envs/extraction/AutismBrainSVM/SVM/parsed_data'
groups = ['autistic', 'control']
X, y = [], []

def process_file(file_path, group_idx):
    img = image.load_img(file_path)
    features = masker.fit_transform(img)
    if features.shape[0] > 1:
        features = np.mean(features, axis=0)
    else:
        features = features.ravel()
    return features, group_idx

with ThreadPoolExecutor() as executor:
    futures = []
    for i, group in enumerate(groups):
        group_path = os.path.join(data_path, group)
        for file_path in glob.glob(os.path.join(group_path, '*.nii.gz')):
            futures.append(executor.submit(process_file, file_path, i))

    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        features, group_idx = future.result()
        X.append(features)
        y.append(group_idx)

X = np.array(X)
y = np.array(y)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the SVM classifier with the best hyperparameters
svm = SVC(C=100, kernel='linear', shrinking=True, class_weight=None)
svm.fit(X_train, y_train)

# Make predictions on the test data
y_pred = svm.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_mat}")
