import os
import numpy as np
from analyze_preprocessed_fmri import analyze_preprocessed_fmri
from feature_extraction import extract_features
from brain_classifier import run_classifier
from concurrent.futures import ProcessPoolExecutor

**ASK CHATGPT: In main.py,how can I define the input_files variable using the proper results from previous scripts? Is there a way to automate this easily, without it being computationally expensive?**

# Replace with the list of input files and their corresponding labels
input_files = [...]
labels = np.array([...])

# Output directory
output_dir = "/path/to/output_dir"

# Process files in parallel using multiple CPU cores
with ProcessPoolExecutor() as executor:
    correlation_matrices = list(executor.map(analyze_preprocessed_fmri, input_files, [output_dir] * len(input_files)))

# Save correlation matrices to disk
np.save(os.path.join(output_dir, "correlation_matrices.npy"), correlation_matrices)

# Extract features from the correlation matrices
features = extract_features(correlation_matrices)

# Run the classifier
run_classifier(features, labels)
