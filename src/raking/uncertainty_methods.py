"""Module with methods to propagate the uncertainties through the raking process"""

def delta_method(dh_x, dh_y, sigma_xx, sigma_yy, sigma_xy):
    """
    """
    dh = np.concatenate((dh_x, dh_y), axis=0)
    sigma = np.concatenate(( \
        np.concatenate((sigma_xx, sigma_xy), axis=1), \
        np.concatenate((np.transpose(sigma_xy), sigma_yy), axis=1)), axis=0)
    variance = np.matmul(np.transpose(dh), np.matmul(sigma, dh))
    return variance

def compute_gradient(
    beta_0: np.ndarray,
    lambda_0: np.ndarray,
    y: np.ndarray,
    A: np.ndarray,
    method: str,
    alpha: float = 1,
    l: np.ndarray = None,
    h: np.ndarray = None, 
    q: np.ndarray = None
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the gradient dbeta/dy and dbeta/ds.

    The covariance matrix of the raked values is phi' Sigma phi'T
    where phi' is the matrix of the partial derivatives of the raked values beta
    with respect to the observations y and margins s. This function computes phi'

    Parameters
    ----------
    beta_0 : np.ndarray
        Vector of raked values
    lambda_0 : np.ndarray
        Corresponding dual
    y : np.ndarray
        Vector of observations
    A : np.ndarray
        Constraints matrix (output of a function from the compute_constraints module)
    method : string
        Raking method (one of chi2, entropic, general, logit)
    alpha : float
        Parameter of the distance function, alpha=1 is the chi2 distance, alpha=0 is the entropic distance
    l : np.ndarray
        Lower bounds for the observations
    h : np.ndarray
        Upper bounds for the observations
    q :  np.ndarray
        Vector of weights (default to all 1)

    Returns
    -------
    Dphi_y : np.ndarray
        Derivatives with respect to the observations
    Dphi_s: np.ndarray
        Derivatives with respect to the margins
    """
    assert isinstance(beta_0, np.ndarray), \
        'The vector of raked values should be a Numpy array.'
    assert len(beta_0.shape) == 1, \
        'The vector of raked values should be a 1D Numpy array.'
    assert isinstance(lambda_0, np.ndarray), \
        'The dual vector should be a Numpy array.'
    assert len(lambda_0.shape) == 1, \
        'The vdual vector should be a 1D Numpy array.'
    assert isinstance(y, np.ndarray), \
        'The vector of observations should be a Numpy array.'
    assert len(y.shape) == 1, \
        'The vector of observations should be a 1D Numpy array.'
    assert len(y) == len(beta_0), \
        'The vectors of observations and raked values should have the same length.'
    assert isinstance(A, np.ndarray), \
        'The constraint matrix should be a Numpy array.'
    assert len(A.shape) == 2, \
        'The constraints matrix should be a 2D Numpy array.'
    assert np.shape(A)[0] == len(lambda_0), \
        'The number of linear constraints should be equal to the length of the dual vector.'
    assert np.shape(A)[1] == len(y), \
        'The number of coefficients for the linear constraints should be equal to the number of observations.'
    assert method in ['chi2', 'entropic', 'general', 'logit'], \
        'The raking method must be "chi2", "entropic", "general", or "logit".'
    if method == 'general':
        assert isinstance(alpha, (int, float)), \
            'The parameter of the distance function should be an integer or a float.'
    if method == 'logit':
        if l is None:
            l = np.zeros(len(y))
        assert isinstance(l, np.ndarray), \
            'The vector of lower bounds should be a Numpy array.'
        assert len(l.shape) == 1, \
            'The vector of lower bounds should be a 1D Numpy array.'
        assert len(y) == len(l), \
            'Observations and lower bounds vectors should have the same length.'
        assert np.all(l >= 0.0), \
            'The lower bounds must be positive.'
        assert np.all(l <= y), \
            'The observations must be superior or equal to the corresponding lower bounds.'
        if h is None:
            h = np.ones(len(y))
        assert isinstance(h, np.ndarray), \
            'The vector of upper bounds should be a Numpy array.'
        assert len(h.shape) == 1, \
            'The vector of upper bounds should be a 1D Numpy array.'
        assert len(y) == len(h), \
            'Observations and upper bounds vectors should have the same length.'
        assert np.all(h > 0.0), \
            'The upper bounds must be strictly positive.'
        assert np.all(h >= y), \
            'The observations must be inferior or equal to the correspondings upper bounds.'   
        assert np.all(l < h), \
            'The lower bounds must be stricty inferior to the correspondings upper bounds.'
    if q is not None:
        assert isinstance(q, np.ndarray), \
            'The vector of weights should be a Numpy array.'
        assert len(q.shape) == 1, \
            'The vector of weights should be a 1D Numpy array.'
        assert len(y) == len(q), \
            'Observations and weights vectors should have the same length.'

    if q is None:
        q = np.ones(len(y))

    # Partial derivatives of the distance function with respect to raked values and observations
    if method == 'chi2':
        DF1_beta_diag = np.zeros(len(beta_0))
        DF1_beta_diag[y!=0] = 1.0 / (q[y!=0] * y[y!=0])
        DF1_beta_diag[y==0] = 0.0
        DF1_beta = np.diag(DF1_beta_diag)
        DF1_y_diag = np.zeros(len(y))
        DF1_y_diag[y!=0] = - beta_0[y!=0] / (q[y!=0] * np.square(y[y!=0]))
        DF1_y_diag[y==0] = 0.0
        DF1_y = np.diag(DF1_y_diag)
    elif method == 'entropic':
        DF1_beta_diag = np.zeros(len(beta_0))
        DF1_beta_diag[beta_0!=0] = 1.0 / (q[beta_0!=0] * beta_0[beta_0!=0])
        DF1_beta_diag[beta_0==0] = 0.0
        DF1_beta = np.diag(DF1_beta_diag)
        DF1_y_diag = np.zeros(len(y))
        DF1_y_diag[y!=0] = - 1.0 / (q[y!=0] * y[y!=0])
        DF1_y_diag[y==0] = 0.0
        DF1_y = np.diag(DF1_y_diag)
    elif method == 'general':
        DF1_beta_diag = np.zeros(len(beta_0))
        DF1_beta_diag[(y!=0)&(beta_0!=0)] = \
            np.power(beta_0[(y!=0)&(beta_0!=0)], alpha - 1.0) / \
            (q[(y!=0)&(beta_0!=0)] * np.power(y[(y!=0)&(beta_0!=0)], alpha))
        DF1_beta_diag[(y==0)|(beta_0==0)] = 0.0
        DF1_beta = np.diag(DF1_beta_diag)
        DF1_y_diag = np.zeros(len(y))
        DF1_y_diag[(y!=0)&(beta_0!=0)] = \
            - np.power(beta_0[(y!=0)&(beta_0!=0)], alpha) / \
            (q[(y!=0)&(beta_0!=0)] * np.power(y[(y!=0)&(beta_0!=0)], alpha + 1.0))
        DF1_y_diag[(y==0)|(beta_0==0)] = 0.0
        DF1_y = np.diag(DF1_y_diag)
    elif method == 'logit':
        DF1_beta_diag = np.zeros(len(beta_0))
        DF1_beta_diag[(beta_0!=l)&(beta_0!=h)] = \
            1.0 / (beta_0[(beta_0!=l)&(beta_0!=h)] - l[(beta_0!=l)&(beta_0!=h)]) + \
            1.0 / (h[(beta_0!=l)&(beta_0!=h)] - beta_0[(beta_0!=l)&(beta_0!=h)])
        DF1_beta_diag[(beta_0==l)|(beta_0==h)] = 0.0
        DF1_beta = np.diag(DF1_beta_diag)
        DF1_y_diag = np.zeros(len(y))
        DF1_y_diag[(y!=l)&(y!=h)] = \
            - 1.0 / (y[(y!=l)&(y!=h)] - l[(y!=l)&(y!=h)]) - \
            1.0 / (h[(y!=l)&(y!=h)] - y[(y!=l)&(y!=h)])
        DF1_y_diag[(y==l)|(y==h)] = 0.0
        DF1_y = np.diag(DF1_y_diag)

    # Gradient with respect to beta and lambda
    DF1_lambda = np.transpose(np.copy(A))
    DF2_beta = np.copy(A)
    DF2_lambda = np.zeros((np.shape(A)[0], np.shape(A)[0]))    
    DF_beta_lambda = np.concatenate(( \
        np.concatenate((DF1_beta, DF1_lambda), axis=1), \
        np.concatenate((DF2_beta, DF2_lambda), axis=1)), axis=0)

    # Gradient with respect to y and s
    DF1_s = np.zeros((np.shape(A)[1], np.shape(A)[0]))
    DF2_y = np.zeros((np.shape(A)[0], np.shape(A)[1]))
    DF2_s = - np.identity(np.shape(A)[0])    
    DF_y_s = np.concatenate(( \
        np.concatenate((DF1_y, DF1_s), axis=1), \
        np.concatenate((DF2_y, DF2_s), axis=1)), axis=0)

    # Solve system DF_beta_lambda Dphi_y_s = - DF_y_s
    Dphi_y_s = np.zeros_like(DF_y_s)
    lu, piv = lu_factor(DF_beta_lambda)
    for i in range(0, np.shape(DF_y_s)[1]):
        Dphi_y_s[:, i] = - lu_solve((lu, piv), DF_y_s[:, i])
        
    # Return gradient of beta and lambda with respect to y and s
    Dphi_y = Dphi_y_s[0:np.shape(A)[1], 0:np.shape(A)[1]]
    Dphi_s = Dphi_y_s[0:np.shape(A)[1], np.shape(A)[1]:(np.shape(A)[0] + np.shape(A)[1])]
    return (Dphi_y, Dphi_s)

