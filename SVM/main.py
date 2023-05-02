from analyze_preprocessed_fmri import analyze_preprocessed_fmri
import os
import glob
import numpy as np
from concurrent.futures import ProcessPoolExecutor

# Set the input and output directories
control_dir = '/path/to/control/fmri_data'
autistic_dir = '/path/to/autistic/fmri_data'
output_dir = '/path/to/output_dir'

# Get the list of input files
control_files = sorted(glob.glob(os.path.join(control_dir, "*.nii.gz")))
autistic_files = sorted(glob.glob(os.path.join(autistic_dir, "*.nii.gz")))

# Define a function to process a single file and return its label and correlation matrix
def process_file(file_path, group_label, output_dir):
    fisher_z_matrix = analyze_preprocessed_fmri(file_path, output_dir)
    return group_label, fisher_z_matrix

# Process all files in parallel using concurrent.futures
correlation_matrices = []
labels = []

with ProcessPoolExecutor() as executor:
    control_results = executor.map(process_file, control_files, ['control'] * len(control_files), [output_dir] * len(control_files))
    autistic_results = executor.map(process_file, autistic_files, ['autistic'] * len(autistic_files), [output_dir] * len(autistic_files))

    for group_label, fisher_z_matrix in control_results:
        labels.append(group_label)
        correlation_matrices.append(fisher_z_matrix)

    for group_label, fisher_z_matrix in autistic_results:
        labels.append(group_label)
        correlation_matrices.append(fisher_z_matrix)

# Save the correlation matrices and labels to disk
np.save(os.path.join(output_dir, 'correlation_matrices.npy'), correlation_matrices)
np.save(os.path.join(output_dir, 'labels.npy'), labels)
