from numpy import ascontiguousarray, atleast_2d


def normalise_outcome(y):
    return y


def normalise_covariates(M):
    M = ascontiguousarray(M, float)
    M = atleast_2d(M.T).T
    return M


def normalise_covariance(K):
    K = ascontiguousarray(K, float)
    if K.ndim != 2:
        raise ValueError("Covariance matrix must have two dimensions.")
    if K.shape[0] != K.shape[1]:
        raise ValueError("Covariance matrix must be square.")
    return K
