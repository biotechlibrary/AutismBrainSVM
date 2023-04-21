import os
import numpy as np
import pandas as pd
from nilearn import datasets, input_data, image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the atlas
atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
masker = input_data.NiftiLabelsMasker(labels_img=atlas.maps, standardize=True)

# Prepare the data
data_path = 'parsed_data'
groups = ['autistic', 'control']
X, y = [], []

for i, group in enumerate(groups):
    group_path = os.path.join(data_path, group)
    for file in os.listdir(group_path):
        file_path = os.path.join(group_path, file)
        if file.endswith('.nii.gz'):
            img = image.load_img(file_path)
            features = masker.fit_transform(img)
            X.append(np.mean(features, axis=0))
            y.append(i)

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
