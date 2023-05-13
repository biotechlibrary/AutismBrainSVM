def extract_features(correlation_matrices, n_components=None):
    """
    This function uses Principal Component Analysis (PCA) to extract features from a stack of correlation matrices.
    If the number of components is not specified, it will be set to one less than the minimum dimension of the stacked matrices.

    Parameters
    ----------
    correlation_matrices : list of np.array
        A list of 2D numpy arrays, where each array represents a correlation matrix.

    n_components : int, optional
        The number of principal components to keep. If not specified, it will be set to one less than the minimum dimension
        of the stacked matrices.

    Returns
    -------
    pca_features : np.array
        The features extracted using PCA. Each row corresponds to the features extracted from one correlation matrix.
    """
    stacked_matrices = np.vstack(correlation_matrices)

    if n_components is None:
        n_components = min(stacked_matrices.shape) - 1

    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(stacked_matrices)

    return pca_features
