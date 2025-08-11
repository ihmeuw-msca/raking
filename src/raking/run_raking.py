"""Module to run the raking problems"""

import numpy as np
import pandas as pd

from raking.compute_constraints import (
    constraints_1D,
    constraints_2D,
    constraints_3D,
    constraints_USHD,
    constraints_USHD_lower,
)
from raking.compute_covariance import compute_covariance_obs
from raking.compute_covariance import (
    compute_covariance_margins_1D,
    compute_covariance_margins_2D,
    compute_covariance_margins_3D,
    compute_covariance_margins_USHD,
)
from raking.compute_covariance import (
    compute_covariance_obs_margins_1D,
    compute_covariance_obs_margins_2D,
    compute_covariance_obs_margins_3D,
)
from raking.compute_covariance import check_covariance
from raking.formatting_methods import (
    format_data_1D,
    format_data_2D,
    format_data_3D,
    format_data_USHD,
    format_data_USHD_lower,
)
from raking.raking_methods import (
    raking_chi2,
    raking_entropic,
    raking_general,
    raking_logit,
)
from raking.uncertainty_methods import compute_covariance, compute_gradient

pd.options.mode.chained_assignment = None


def run_raking(
    dim: int | str,
    df_obs: pd.DataFrame,
    df_margins: list,
    var_names: list | None,
    margin_names: list = None,
    draws: str = "draws",
    cov_mat: bool = True,
    sigma_yy: np.ndarray = None,
    sigma_ss: np.ndarray = None,
    sigma_ys: np.ndarray = None,
    method: str = "chi2",
    alpha: float = 1,
    weights: str = None,
    lower: str = None,
    upper: str = None,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    tol: float = 1.0e-11,
    gamma: float = 1.0e-4,
    max_iter: int = 500,
) -> np.ndarray:
    """
    This function allows the user to run the raking problem.

    Parameters
    ----------
    dim : integer or string
        Dimension of the raking problem (1, 2, 3) or special case (USHD)
    df_obs : pd.DataFrame
        Observations data
    df_margins : list of pd.DataFrame
        list of data frames contatining the margins data
    var_names : list of strings
        Names of the variables over which we rake (e.g. cause, race, county). None if using special case.
    margin_names : list
        Names for the all causes, all races, all counties categories (length 3). None if using 1D, 2D or 3D raking.
    draws: string
        Name of the column that contains the samples.
    cov_mat : boolean
        If True, compute the covariance matrix of the raked values
    sigma_yy: np.ndarray
        Covariance matrix of the observations. We assume that there are sorted by var3, var2, var1.
        If None, the observation data frame must contain samples and we compte the sample covariance matrix.
    sigma_ss: np.ndarray
        Covariance matrix of the margins.
        If None, the margins data frames must contain samples and we compte the sample covariance matrix.
    sigma_ys: np.ndarray
        Covariance matrix of the observations and the margins.
        If None, the observations and margins data frames must contain samples and we compte the sample covariance matrix.
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
    tol: float
        Tolerance for the convergence
    gamma : float
        Parameter for Armijo rule
    max_iter : int
        Number of iterations for Newton's root finding method

    Returns
    -------
    df_obs : pd.DataFrame
        The initial observations data frame with an additional column for the raked values
    """
    assert isinstance(dim, int) or isinstance(dim, str), (
        "The dimension of the raking problem must be an integer or string."
    )
    assert dim in [
        1,
        2,
        3,
        "USHD",
        "USHD_lower",
    ], (
        "The dimension of the raking problem must be 1, 2, 3, USHD or USHD_lower."
    )
    assert isinstance(cov_mat, bool), (
        "cov_mat indicates whether we compute the covariance matrix, must be True or False."
    )
    if dim in [1, 2, 3]:
        assert isinstance(var_names, list), (
            "The variables over which we rake must be entered as a list."
        )
        assert dim == len(var_names), (
            "The number of variables over which we rake must be equal to the dimension of the problem."
        )
    else:
        var_names = ["cause", "race", "county"]
    if dim in ["USHD", "USHD_lower"]:
        assert isinstance(margin_names, list), (
            "Please enter the names of the all causes, all races, all counties categories as a list."
        )
        assert len(margin_names) == 3, (
            "There should be a margin name for each of the three variables cause, race, and county."
        )
    else:
        margin_names = None
    assert isinstance(df_margins, list), (
        "The margins data frames must be entered as a list."
    )
    if dim in [1, 2, 3]:
        assert dim == len(df_margins), (
            "The number of margins data frames must be equal to the dimension of the problem."
        )
    elif dim == "USHD":
        assert len(df_margins) == 1, (
            "There should be only one margins data frame in the list."
        )
    else:
        assert len(df_margins) == 3, (
            "There should be three margins data frames in the list."
        )
    assert isinstance(method, str), (
        "The name of the distance function used for the raking must be a string."
    )
    assert method in [
        "chi2",
        "entropic",
        "general",
        "logit",
    ], "The distance function must be chi2, entropic, general or logit."

    df_obs = df_obs.copy(deep=True)

    # Compute the covariance matrix
    if cov_mat:
        if dim == 1:
            (sigma_yy, sigma_ss, sigma_ys) = compute_covariance_1D(
                df_obs,
                df_margins,
                var_names,
                draws,
                sigma_yy,
                sigma_ss,
                sigma_ys,
            )
        elif dim == 2:
            (sigma_yy, sigma_ss, sigma_ys) = compute_covariance_2D(
                df_obs,
                df_margins,
                var_names,
                draws,
                sigma_yy,
                sigma_ss,
                sigma_ys,
            )
        elif dim == 3:
            (sigma_yy, sigma_ss, sigma_ys) = compute_covariance_3D(
                df_obs,
                df_margins,
                var_names,
                draws,
                sigma_yy,
                sigma_ss,
                sigma_ys,
            )
        elif dim == "USHD":
            (sigma_yy, sigma_ss, sigma_ys) = compute_covariance_USHD(
                df_obs,
                df_margins,
                var_names,
                draws,
                sigma_yy,
                sigma_ss,
                sigma_ys,
            )
        else:
            pass
        # Check if matrix is definite positive
        (sigma_yy, sigma_ss, sigma_ys) = check_covariance(
            sigma_yy, sigma_ss, sigma_ys, rtol, atol
        )

    # Compute the mean (if we have draws)
    if cov_mat:
        if dim in [1, 2, 3]:
            (df_obs, df_margins) = compute_mean(
                df_obs, df_margins, var_names, draws
            )
        else:
            (df_obs, df_margins) = compute_mean(
                df_obs, df_margins, ["race_county"], draws
            )

    # Get the input variables for the raking
    if dim == 1:
        (y, s, q, l, h, A) = run_raking_1D(
            df_obs, df_margins, var_names, weights, lower, upper, rtol, atol
        )
    elif dim == 2:
        (y, s, q, l, h, A) = run_raking_2D(
            df_obs, df_margins, var_names, weights, lower, upper, atol, rtol
        )
    elif dim == 3:
        (y, s, q, l, h, A) = run_raking_3D(
            df_obs, df_margins, var_names, weights, lower, upper, rtol, atol
        )
    elif dim == "USHD":
        (y, s, q, l, h, A) = run_raking_USHD(
            df_obs, df_margins, weights, lower, upper, rtol, atol
        )
    elif dim == "USHD_lower":
        (y, s, q, l, h, A) = run_raking_USHD_lower(
            df_obs, df_margins, margin_names, weights, lower, upper, rtol, atol
        )
    else:
        pass

    # Rake
    if method == "chi2":
        (beta, lambda_k) = raking_chi2(y, A, s, q)
    elif method == "entropic":
        (beta, lambda_k, iter_eps) = raking_entropic(
            y, A, s, q, tol, gamma, max_iter
        )
    elif method == "general":
        (beta, lambda_k, iter_eps) = raking_general(
            y, A, s, alpha, q, tol, gamma, max_iter
        )
    elif method == "logit":
        (beta, lambda_k, iter_eps) = raking_logit(
            y, A, s, l, h, q, tol, 1.0, max_iter
        )
    else:
        pass

    # Create data frame for the raked values
    var_names.reverse()
    df_obs.sort_values(by=var_names, inplace=True)
    df_obs["raked_value"] = beta

    # Compute the covariance matrix of the raked values
    if cov_mat:
        (Dphi_y, Dphi_s) = compute_gradient(
            beta, lambda_k, y, A, method, alpha, l, h, q
        )
        sigma = compute_covariance(Dphi_y, Dphi_s, sigma_yy, sigma_ss, sigma_ys)
        df_obs["variance"] = np.diag(sigma)
    else:
        Dphi_y = None
        Dphi_s = None
        sigma = None
    return (df_obs, Dphi_y, Dphi_s, sigma)


def run_raking_1D(
    df_obs: pd.DataFrame,
    df_margins: list,
    var_names: list,
    weights: str = None,
    lower: str = None,
    upper: str = None,
    rtol: float = 1e-05,
    atol: float = 1e-08,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray,
]:
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
    (y, s, I, q, l, h) = format_data_1D(
        df_obs, df_margins, var_name, weights, lower, upper
    )
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
    atol: float = 1e-08,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray,
]:
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
    (y, s1, s2, I, J, q, l, h) = format_data_2D(
        df_obs, df_margins_1, df_margins_2, var_names, weights, lower, upper
    )
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
    atol: float = 1e-08,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray,
]:
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
    (y, s1, s2, s3, I, J, K, q, l, h) = format_data_3D(
        df_obs,
        df_margins_1,
        df_margins_2,
        df_margins_3,
        var_names,
        weights,
        lower,
        upper,
    )
    (A, s) = constraints_3D(s1, s2, s3, I, J, K, rtol, atol)
    return (y, s, q, l, h, A)


def run_raking_USHD(
    df_obs: pd.DataFrame,
    df_margins: list,
    margin_names: list,
    weights: str = None,
    lower: str = None,
    upper: str = None,
    rtol: float = 1e-05,
    atol: float = 1e-08,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray,
]:
    """
    This function prepares variables to run the raking problem for the USHD case.

    Parameters
    ----------
    df_obs : pd.DataFrame
        Observations data
    df_margins : list of pd.DataFrame
        list of data frames contatining the margins data
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
    (y, s, I, J, K, q, l, h) = format_data_USHD(
        df_obs,
        df_margins,
        margin_names,
        weights,
        lower,
        upper,
    )
    (A, s) = constraints_USHD(s, I, J, K, rtol, atol)
    return (y, s, q, l, h, A)


def run_raking_USHD_lower(
    df_obs: pd.DataFrame,
    df_margins: list,
    margin_names: list,
    weights: str = None,
    lower: str = None,
    upper: str = None,
    rtol: float = 1e-05,
    atol: float = 1e-08,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray,
]:
    """
    This function prepares variables to run the raking problem for the USHD case (lower levels).

    Parameters
    ----------
    df_obs : pd.DataFrame
        Observations data
    df_margins : list of pd.DataFrame
        list of data frames contatining the margins data
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
    df_margins_cause = df_margins[0]
    df_margins_county = df_margins[1]
    df_margins_all_causes = df_margins[2]
    (y, s_cause, s_county, s_all_causes, I, J, K, q, l, h) = (
        format_data_USHD_lower(
            df_obs,
            df_margins_cause,
            df_margins_county,
            df_margins_all_causes,
            margin_names,
            weights,
            lower,
            upper,
        )
    )
    (A, s) = constraints_USHD_lower(
        s_cause, s_county, s_all_causes, I, J, K, rtol, atol
    )
    return (y, s, q, l, h, A)


def compute_covariance_1D(
    df_obs: pd.DataFrame,
    df_margins: pd.DataFrame,
    var_names: list,
    draws: str,
    sigma_yy: np.ndarray = None,
    sigma_ss: np.ndarray = None,
    sigma_ys: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the covariance matrix of observations and margins in 1D.

    Parameters
    ----------
    df_obs : pd.DataFrame
        Observations data
    df_margins : list of pd.DataFrame
        list of data frames contatining the margins data
    var_names : list of strings
        Names of the variables over which we rake (e.g. cause, race, county)
    draws: string
        Name of the column that contains the samples.
    sigma_yy : np.ndarray
        Covariance matrix of the observations
    sigma_ss : np.ndarray
        Covariance matrix of the margins
    sigma_ys : np.ndarray
        Covariance matrix of the observations and margins

    Returns
    -------
    sigma_yy : np.ndarray
        Covariance matrix of the observations
    sigma_ss : np.ndarray
        Covariance matrix of the margins
    sigma_ys : np.ndarray
        Covariance matrix of the observations and margins
    """
    df_margins = df_margins[0]
    if sigma_yy is None:
        sigma_yy = compute_covariance_obs(df_obs, var_names, draws)
    if sigma_ss is None:
        sigma_ss = compute_covariance_margins_1D(df_margins, var_names, draws)
    if sigma_ys is None:
        sigma_ys = compute_covariance_obs_margins_1D(
            df_obs, df_margins, var_names, draws
        )
    return (sigma_yy, sigma_ss, sigma_ys)


def compute_covariance_2D(
    df_obs: pd.DataFrame,
    df_margins: pd.DataFrame,
    var_names: list,
    draws: str,
    sigma_yy: np.ndarray,
    sigma_ss: np.ndarray,
    sigma_ys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the covariance matrix of observations and margins in 2D.

    Parameters
    ----------
    df_obs : pd.DataFrame
        Observations data
    df_margins : list of pd.DataFrame
        list of data frames contatining the margins data
    var_names : list of strings
        Names of the variables over which we rake (e.g. cause, race, county)
    draws: string
        Name of the column that contains the samples.
    sigma_yy : np.ndarray
        Covariance matrix of the observations
    sigma_ss : np.ndarray
        Covariance matrix of the margins
    sigma_ys : np.ndarray
        Covariance matrix of the observations and margins

    Returns
    -------
    sigma_yy : np.ndarray
        Covariance matrix of the observations
    sigma_ss : np.ndarray
        Covariance matrix of the margins
    sigma_ys : np.ndarray
        Covariance matrix of the observations and margins
    """
    df_margins_1 = df_margins[0]
    df_margins_2 = df_margins[1]
    if sigma_yy is None:
        sigma_yy = compute_covariance_obs(df_obs, var_names, draws)
    if sigma_ss is None:
        sigma_ss = compute_covariance_margins_2D(
            df_margins_1, df_margins_2, var_names, draws
        )
    if sigma_ys is None:
        sigma_ys = compute_covariance_obs_margins_2D(
            df_obs, df_margins_1, df_margins_2, var_names, draws
        )
    return (sigma_yy, sigma_ss, sigma_ys)


def compute_covariance_3D(
    df_obs: pd.DataFrame,
    df_margins: pd.DataFrame,
    var_names: list,
    draws: str,
    sigma_yy: np.ndarray,
    sigma_ss: np.ndarray,
    sigma_ys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the covariance matrix of observations and margins in 3D.

    Parameters
    ----------
    df_obs : pd.DataFrame
        Observations data
    df_margins : list of pd.DataFrame
        list of data frames contatining the margins data
    var_names : list of strings
        Names of the variables over which we rake (e.g. cause, race, county)
    draws: string
        Name of the column that contains the samples.
    sigma_yy : np.ndarray
        Covariance matrix of the observations
    sigma_ss : np.ndarray
        Covariance matrix of the margins
    sigma_ys : np.ndarray
        Covariance matrix of the observations and margins

    Returns
    -------
    sigma_yy : np.ndarray
        Covariance matrix of the observations
    sigma_ss : np.ndarray
        Covariance matrix of the margins
    sigma_ys : np.ndarray
        Covariance matrix of the observations and margins
    """
    df_margins_1 = df_margins[0]
    df_margins_2 = df_margins[1]
    df_margins_3 = df_margins[2]
    if sigma_yy is None:
        sigma_yy = compute_covariance_obs(df_obs, var_names, draws)
    if sigma_ss is None:
        sigma_ss = compute_covariance_margins_3D(
            df_margins_1, df_margins_2, df_margins_3, var_names, draws
        )
    if sigma_ys is None:
        sigma_ys = compute_covariance_obs_margins_3D(
            df_obs, df_margins_1, df_margins_2, df_margins_3, var_names, draws
        )
    return (sigma_yy, sigma_ss, sigma_ys)


def compute_covariance_USHD(
    df_obs: pd.DataFrame,
    df_margins: pd.DataFrame,
    var_names: list,
    draws: str,
    sigma_yy: np.ndarray,
    sigma_ss: np.ndarray,
    sigma_ys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the covariance matrix of observations and margins for the USHD case.

    Parameters
    ----------
    df_obs : pd.DataFrame
        Observations data
    df_margins : list of pd.DataFrame
        list of data frames contatining the margins data
    var_names : list of strings
        Names of the variables over which we rake (cause, race, county)
    draws: string
        Name of the column that contains the samples.
    sigma_yy : np.ndarray
        Covariance matrix of the observations
    sigma_ss : np.ndarray
        Covariance matrix of the margins
    sigma_ys : np.ndarray
        Covariance matrix of the observations and margins

    Returns
    -------
    sigma_yy : np.ndarray
        Covariance matrix of the observations
    sigma_ss : np.ndarray
        Covariance matrix of the margins
    sigma_ys : np.ndarray
        Covariance matrix of the observations and margins
    """
    df_margins = df_margins[0]
    if sigma_yy is None:
        sigma_yy = compute_covariance_obs(df_obs, var_names, draws)
    if sigma_ss is None:
        I = len(df_obs["cause"].unique()) - 1
        J = len(df_obs["race"].unique()) - 1
        K = len(df_obs["county"].unique())
        sigma_ss = compute_covariance_margins_USHD(
            df_margins, ["cause"], draws, I, J, K
        )
    if sigma_ys is None:
        sigma_ys = np.zeros((sigma_yy.shape[0], sigma_ss.shape[0]))
    return (sigma_yy, sigma_ss, sigma_ys)


def compute_mean(
    df_obs: pd.DataFrame, df_margins: list, var_names: list, draws: str
) -> tuple[pd.DataFrame, list]:
    """Compute the means of the values over all the samples.

    Parameters
    ----------
    df_obs : pd.DataFrame
        Observations data
    df_margins : list of pd.DataFrame
        list of data frames contatining the margins data
    var_names : list of strings
        Names of the variables over which we rake (e.g. cause, race, county)
    draws: string
        Name of the column that contains the samples.

    Returns
    -------
    df_obs_mean : pd.DataFrame
        Means of observations data
    df_margins_mean : list of pd.DataFrame
        list of data frames contatining the mans of the margins data
    """
    columns = df_obs.columns.drop([draws, "value"]).to_list()
    df_obs_mean = (
        df_obs.groupby(columns).mean().reset_index().drop(columns=[draws])
    )
    df_margins_mean = []
    for df_margin, var_name in zip(df_margins, var_names):
        value_name = "value_agg_over_" + var_name
        columns = df_margin.columns.drop([draws, value_name]).to_list()
        if len(columns) == 0:
            df_margin_mean = pd.DataFrame(
                {value_name: np.array([df_margin.mean()[value_name]])}
            )
        else:
            df_margin_mean = (
                df_margin.groupby(columns)
                .mean()
                .reset_index()
                .drop(columns=[draws])
            )
        df_margins_mean.append(df_margin_mean)
    return (df_obs_mean, df_margins_mean)
