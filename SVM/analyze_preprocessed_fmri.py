import os
import numpy as np
from nilearn.connectome import ConnectivityMeasure
from nilearn.input_data import NiftiLabelsMasker

def analyze_preprocessed_fmri(input_file, atlas):
    """
    This function computes the functional connectivity matrix of the preprocessed fMRI data. 
    It first extracts the time series from the ROIs defined by the atlas, then calculates the 
    functional connectivity matrix, applies Fisher z-transformation to normalize the data, and 
    replaces any NaN values with the mean of the matrix.

    Parameters
    ----------
    input_file : str
        Path to the preprocessed fMRI data in NIfTI format.
    atlas : str
        Path to the brain atlas in NIfTI format used for defining ROIs.

    Returns
    -------
    fisher_z_matrix : np.array
        A 2D numpy array representing the Fisher z-transformed functional connectivity matrix.

    """
    masker = NiftiLabelsMasker(labels_img=atlas, standardize=False)
    roi_time_series = masker.fit_transform(input_file)

    correlation_measure = ConnectivityMeasure(kind='correlation')
    fc_matrix = correlation_measure.fit_transform([roi_time_series])[0]
    np.fill_diagonal(fc_matrix, 0.9999)
    fisher_z_matrix = np.arctanh(fc_matrix)

    nan_mean = np.nanmean(fisher_z_matrix)
    fisher_z_matrix = np.where(np.isnan(fisher_z_matrix), nan_mean, fisher_z_matrix)

    return fisher_z_matrix
