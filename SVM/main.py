import os
import glob
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from analyze_preprocessed_fmri import analyze_preprocessed_fmri
from feature_extraction import extract_features
from brain_classifier import train_and_evaluate_classifier
from nilearn.input_data import NiftiLabelsMasker



control_dir = '/home/pau/micromamba/envs/extraction/AutismBrainSVM/SVM/parsed_data/control'
autistic_dir = '/home/pau/micromamba/envs/extraction/AutismBrainSVM/SVM/parsed_data/autistic'
atlas_path = "/home/pau/Documents/Research/thesis/RSN_atlas_3mm.nii.gz"

control_files = sorted(glob.glob(os.path.join(control_dir, "*.nii.gz")))
autistic_files = sorted(glob.glob(os.path.join(autistic_dir, "*.nii.gz")))

def process_file(args):
    file_path, group_label = args
    fisher_z_matrix = analyze_preprocessed_fmri(file_path, atlas_path)
    return group_label, fisher_z_matrix



correlation_matrices = []
labels = []

with ProcessPoolExecutor(max_workers=5) as executor:
    control_results = executor.map(process_file, zip(control_files, ['control'] * len(control_files)))
    autistic_results = executor.map(process_file, zip(autistic_files, ['autistic'] * len(autistic_files)))

    for group_label, fisher_z_matrix in control_results:
        labels.append(group_label)
        correlation_matrices.append(fisher_z_matrix)

    for group_label, fisher_z_matrix in autistic_results:
        labels.append(group_label)
        correlation_matrices.append(fisher_z_matrix)

# Feature extraction
features = extract_features(correlation_matrices)

# Classifier training and evaluation
train_and_evaluate_classifier(features, labels)
