import numpy as np

def pca(X, ndim):
    """
    Perform PCA on the given data matrix X and reduce its dimensionality to ndim.

    Parameters:
    X (numpy.ndarray): Data matrix with shape (n, d) where n is the number of data points and d is the number of dimensions
    ndim (int): Number of dimensions to reduce to

    Returns:
    T (numpy.ndarray): The transformed data matrix with shape (n, ndim)
    """
    # Center the data by subtracting the mean of each feature
    X_centered = X - np.mean(X, axis=0)
    
    # Perform Singular Value Decomposition
    U, S, Vt = np.linalg.svd(X_centered)
    
    # Select the top 'ndim' components
    Ur = U[:, :ndim]
    Sr = np.diag(S[:ndim])
    
    # Transform the data to the new reduced dimension space
    T = Ur @ Sr
    
    return T
