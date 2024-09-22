"""Module to run the raking problems"""

import numpy as np
import pandas as pd

from raking.compute_constraints import constraints_1D, constraints_2D, constraints_3D
from raking.formatting_methods import format_data_1D, format_data_2D, format_data_3D
from raking.raking_methods import raking_chi2, raking_entropic, raking_general, raking_logit

def run_raking(
    dim: int,
    directory_name: str,
    df_obs: pd.DataFrame,
    df_margins: list,
    var_names: list,
    method: str = 'chi2',
    alpha: float = 1,
    weights: str = None,
    lower: str = None,
    upper: str = None,
    rtol: float = 1e-05, 
    atol:float = 1e-08,
    gamma0: float = 1.0,
    max_iter: int = 500
) -> None:
    """
    This function allows the user to run the raking problem.

    Parameters
    ----------
    dim : integer
        Dimension of the raking problem (1, 2, 3)
    directory_name: string
        Name of the directory where we write the results
    df_obs : pd.DataFrame
        Observations data
    df_margins : list of pd.DataFrame
        list of data frames contatining the margins data
    var_names : list of strings
        Names of the variables over which we rake (e.g. cause, race, county)
    method : string
        Name of the distance function used for the raking.
        Possible values are chi2, entropic, general, logit
    alpha : float
        Parameter of the distance function, alpha=1 is the chi2 distance, alpha=0 is the entropic distance
    weights : string
        Name of the column containing the raking weights
    lower : string
        Name of the column containing the lower boundaries (for logit raking)
    upper : string
        Name of the column containing the upper boundaries (for logit raking)
    rtol : float
        Relative tolerance to check whether the margins are consistant. See numpy.allclose documentation for details.
    atol : float
        Absolute tolerance to check whether the margins are consistant. See numpy.allclose documentation for details.
    gamma0 : float
        Initial value for line search
    max_iter : int
        Number of iterations for Newton's root finding method

    Returns
    -------
    None
    """
    assert isinstance(dim, int), \
        'The dimension of the raking problem must be an integer.'
    assert dim in [1, 2, 3], \
        'The dimension of the raking problem must be 1, 2 or 3.'
    assert isinstance(var_names, list), \
        'The variables over which we rake must be entered as a list.'
    assert dim == len(var_names), \
        'The number of variables over which we rake must be equal to the dimension of the problem.'
    assert isinstance(df_margins, list), \
        'The margins data frames must be entered as a list.'
    assert dim == len(df_margins), \
        'The number of margins data frames must be equal to the dimension of the problem.'
    assert isinstance(method, str), \
        'The name of the distance function used for the raking must be a string.'
    assert method in ['chi2', 'entropic', 'general', 'logit'], \
        'The distance function must be chi2, entropic, general or logit.'
    
    # Get the input variables for the raking
    if dim == 1:
        (y, s, q, l, h, A) = run_raking_1D(df_obs, df_margins, var_names, weights, lower, upper, rtol, atol)
    elif dim == 2:
        (y, s, q, l, h, A) = run_raking_2D(df_obs, df_margins, var_names, weights, lower, upper, atol, rtol)
    elif dim == 3:
        (y, s, q, l, h, A) = run_raking_3D(df_obs, df_margins, var_names, weights, lower, upper, rtol, atol)
    else:
        pass
    # Rake
    if method == 'chi2':
        (beta, lambda_k) = raking_chi2(y, A, s, q)
    elif method == 'entropic':
        (beta, lambda_k, iter_eps) = raking_entropic(y, A, s, q, gamma0, max_iter)
    elif method == 'general':
        (beta, lambda_k, iter_eps) = raking_general(y, A, s, alpha, q, gamma0, max_iter)
    elif method == 'logit':
        (beta, lambda_k, iter_eps) = raking_logit(y, A, s, l, h, q, gamma0, max_iter)
    else:
        pass
    # Write output file
    var_names.reverse()
    df_obs.sort_values(by=var_names, inplace=True)
    df_obs['raked_value'] = beta
    df_obs.to_csv(directory_name + '/raked_observations.csv', index=False)

def run_raking_1D(
    df_obs: pd.DataFrame,
    df_margins: list,
    var_names: list,
    weights: str = None,
    lower: str = None,
    upper: str = None,
    rtol: float = 1e-05,
    atol:float = 1e-08
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray]:
    """
    This function prepares variables to run the raking problem in 1D.

    Parameters
    ----------
    df_obs : pd.DataFrame
        Observations data
    df_margins : list of pd.DataFrame
        list of data frames contatining the margins data
    var_names : list of strings
        Names of the variables over which we rake (e.g. cause, race, county)
    weights : string
        Name of the column containing the raking weights
    lower : string
        Name of the column containing the lower boundaries (for logit raking)
    upper : string
        Name of the column containing the upper boundaries (for logit raking)
    rtol : float
        Relative tolerance to check whether the margins are consistant. See numpy.allclose documentation for details.
    atol : float
        Absolute tolerance to check whether the margins are consistant. See numpy.allclose documentation for details.

    Returns
    -------
    y : np.ndarray
        Vector of observations
    s : np.ndarray
        Margins vector
    q : np.ndarray
        Vector of weights
    l : np.ndarray
        Lower bounds for the observations
    h : np.ndarray
        Upper bounds for the observations
    A : np.ndarray
        Constraints matrix
    """
    df_margins = df_margins[0]
    var_name = var_names[0]
    (y, s, I, q, l, h) = format_data_1D(df_obs, df_margins, var_name, weights, lower, upper)
    (A, s) = constraints_1D(s, I)
    return (y, s, q, l, h, A)

def run_raking_2D(
    df_obs: pd.DataFrame,
    df_margins: list,
    var_names: list,
    weights: str = None,
    lower: str = None,
    upper: str = None,
    rtol: float = 1e-05, 
    atol:float = 1e-08
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray]:
    """
    This function prepares variables to run the raking problem in 2D.

    Parameters
    ----------
    df_obs : pd.DataFrame
        Observations data
    df_margins : list of pd.DataFrame
        list of data frames contatining the margins data
    var_names : list of strings
        Names of the variables over which we rake (e.g. cause, race, county)
    weights : string
        Name of the column containing the raking weights
    lower : string
        Name of the column containing the lower boundaries (for logit raking)
    upper : string
        Name of the column containing the upper boundaries (for logit raking)
    rtol : float
        Relative tolerance to check whether the margins are consistant. See numpy.allclose documentation for details.
    atol : float
        Absolute tolerance to check whether the margins are consistant. See numpy.allclose documentation for details.

    Returns
    -------
    y : np.ndarray
        Vector of observations
    s : np.ndarray
        Margins vector
    q : np.ndarray
        Vector of weights
    l : np.ndarray
        Lower bounds for the observations
    h : np.ndarray
        Upper bounds for the observations
    A : np.ndarray
        Constraints matrix
    """
    df_margins_1 = df_margins[0]
    df_margins_2 = df_margins[1]
    (y, s1, s2, I, J, q, l, h) = format_data_2D(df_obs, df_margins_1, df_margins_2, var_names, weights, lower, upper)
    (A, s) = constraints_2D(s1, s2, I, J, rtol, atol)
    return (y, s, q, l, h, A)

def run_raking_3D(
    df_obs: pd.DataFrame,
    df_margins: list,
    var_names: list,
    weights: str = None,
    lower: str = None,
    upper: str = None,
    rtol: float = 1e-05, 
    atol:float = 1e-08
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray]:
    """
    This function prepares variables to run the raking problem in 3D.

    Parameters
    ----------
    df_obs : pd.DataFrame
        Observations data
    df_margins : list of pd.DataFrame
        list of data frames contatining the margins data
    var_names : list of strings
        Names of the variables over which we rake (e.g. cause, race, county)
    weights : string
        Name of the column containing the raking weights
    lower : string
        Name of the column containing the lower boundaries (for logit raking)
    upper : string
        Name of the column containing the upper boundaries (for logit raking)
    rtol : float
        Relative tolerance to check whether the margins are consistant. See numpy.allclose documentation for details.
    atol : float
        Absolute tolerance to check whether the margins are consistant. See numpy.allclose documentation for details.

    Returns
    -------
    y : np.ndarray
        Vector of observations
    s : np.ndarray
        Margins vector
    q : np.ndarray
        Vector of weights
    l : np.ndarray
        Lower bounds for the observations
    h : np.ndarray
        Upper bounds for the observations
    A : np.ndarray
        Constraints matrix
    """
    df_margins_1 = df_margins[0]
    df_margins_2 = df_margins[1]
    df_margins_3 = df_margins[2]
    (y, s1, s2, s3, I, J, K, q, l, h) = format_data_3D(df_obs, df_margins_1, df_margins_2, df_margins_3, var_names, weights, lower, upper)
    (A, s) = constraints_3D(s1, s2, s3, I, J, K, rtol, atol)
    return (y, s, q, l, h, A)

