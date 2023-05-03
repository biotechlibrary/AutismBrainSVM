import numpy as np
from sklearn.decomposition import PCA

def extract_features(correlation_matrices, n_components=None):
    stacked_matrices = np.vstack(correlation_matrices)
    
    if n_components is None:
        n_components = min(stacked_matrices.shape) - 1

    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(stacked_matrices)

    return pca_features
