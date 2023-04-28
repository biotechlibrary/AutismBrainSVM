import numpy as np
from sklearn.decomposition import PCA

def extract_features(correlation_matrices, n_components=10):
    """
    Extracts features from the correlation matrices using principal component analysis (PCA).

    Args:
        correlation_matrices (list): A list of correlation matrices, one per subject.
        n_components (int): The number of principal components to retain (default=10).

    Returns:
        numpy.ndarray: The extracted features, with shape (n_subjects, n_components).

    Example:
        >>> features = extract_features(correlation_matrices, n_components=20)
    """
    # Stack correlation matrices into a single 2D array
    stacked_matrices = np.vstack(correlation_matrices)

    # Perform PCA
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(stacked_matrices)

    return pca_features
