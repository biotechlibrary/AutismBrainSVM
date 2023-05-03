import os
import numpy as np
from nilearn.connectome import ConnectivityMeasure
from nilearn.input_data import NiftiLabelsMasker


def analyze_preprocessed_fmri(input_file, atlas):
    masker = NiftiLabelsMasker(labels_img=atlas, standardize=False)
    roi_time_series = masker.fit_transform(input_file)

    correlation_measure = ConnectivityMeasure(kind='correlation')
    fc_matrix = correlation_measure.fit_transform([roi_time_series])[0]
    np.fill_diagonal(fc_matrix, 0.9999)
    fisher_z_matrix = np.arctanh(fc_matrix)

    nan_mean = np.nanmean(fisher_z_matrix)
    fisher_z_matrix = np.where(np.isnan(fisher_z_matrix), nan_mean, fisher_z_matrix)


    return fisher_z_matrix
