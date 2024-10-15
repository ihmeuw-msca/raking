"""Module with methods to compute the covaraince matrices of observations and margins"""

import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None

def compute_covariance_obs(
    df_obs: pd.DataFrame,
    var_names: list,
    draws: str
) -> np.ndarray:
    """Compute the covariance matrix of the observations.

    The observations will be sorted by var3, var2, var1, meaning that
    sigma_yy contains on its diagonal in this order the variances of
    y_111, ... , y_I11, y_121, ... , y_IJ1, y_112, ... , y_IJK.

    Parameters
    ----------
    df_obs : pd.DataFrame
        Observations data
    var_names : list of strings
        Names of the variables over which we rake (e.g. cause, race, county)
    draws : string
        Names of the column containing the indices of the draws

    Returns
    -------
    sigma_yy : np.ndarray
        (I * J * K) * (I * J * K) covariance matrix
    """
    nsamples = len(df_obs[draws].unique())
    var_names.reverse()
    df = df_obs[['value'] + var_names + [draws]]
    df.sort_values(by=var_names + [draws], inplace=True)
    value = df['value'].to_numpy()
    X = np.reshape(value, shape=(nsamples, -1), order='F')
    Xmean = np.mean(X, axis=0)
    Xc = X - Xmean
    sigma_yy = np.matmul(np.transpose(Xc), Xc) / nsamples
    return sigma_yy

def compute_covariance_margins_1D(
    df_margins: pd.DataFrame,
    var_names: list,
    draws: str
) -> np.ndarray:
    """Compute the covariance matrix of the margins in 1D.

    Parameters
    ----------
    df_margins : pd.DataFrame
        Margins data (sums over the first variable)
    var_names : list of strings
        Names of the variables over which we rake (e.g. cause, race, county)
    draws : string
        Names of the column containing the indices of the draws

    Returns
    -------
    sigma_ss : np.ndarray
        1 * 1 covariance matrix
    """
    nsamples = len(df_margins[draws].unique())
    df = df_margins[['value_agg_over_' + var_names[0]] + [draws]]
    df.sort_values(by=[draws], inplace=True)
    value = df['value_agg_over_' + var_names[0]].to_numpy()
    X = np.reshape(value, shape=(nsamples, -1), order='F')
    Xmean = np.mean(X, axis=0)
    Xc = X - Xmean
    sigma_ss = np.matmul(np.transpose(Xc), Xc) / nsamples
    return sigma_ss

def compute_covariance_margins_2D(
    df_margins_1: pd.DataFrame,
    df_margins_2: pd.DataFrame,
    var_names: list,
    draws: str
) -> np.ndarray:
    """Compute the covariance matrix of the margins in 2D.

    The margins are sorted in the same order as what is done
    when computing the constraint matrix.

    Parameters
    ----------
    df_margins_1 : pd.DataFrame
        Margins data (sums over the first variable)
    df_margins_2 : pd.DataFrame
        Margins data (sums over the second variable)
    var_names : list of strings
        Names of the variables over which we rake (e.g. cause, race, county)
    draws : string
        Names of the column containing the indices of the draws

    Returns
    -------
    sigma_ss : np.ndarray
        (I + J - 1) * (I + J - 1) covariance matrix
    """
    nsamples = len(df_margins_1[draws].unique())
    df1 = df_margins_1[[var_names[1], 'value_agg_over_' + var_names[0], draws]]
    df1.sort_values(by=[var_names[1], draws], inplace=True)
    df2 = df_margins_2[[var_names[0], 'value_agg_over_' + var_names[1], draws]]
    df2.sort_values(by=[var_names[0], draws], inplace=True)
    value1 = df1['value_agg_over_' + var_names[0]].to_numpy()
    value2 = df2['value_agg_over_' + var_names[1]].to_numpy()
    value = np.concatenate((value1, value2))
    X = np.reshape(value, shape=(nsamples, -1), order='F')
    X = X[:, 0:-1]
    Xmean = np.mean(X, axis=0)
    Xc = X - Xmean
    sigma_ss = np.matmul(np.transpose(Xc), Xc) / nsamples
    return sigma_ss

def compute_covariance_margins_3D(
    df_margins_1: pd.DataFrame,
    df_margins_2: pd.DataFrame,
    df_margins_3: pd.DataFrame,
    var_names: list,
    draws:str
) -> np.ndarray:
    """Compute the covariance matrix of the margins in 3D.

    The margins are sorted in the same order as what is done
    when computing the constraint matrix.

    Parameters
    ----------
    df_margins_1 : pd.DataFrame
        Margins data (sums over the first variable)
    df_margins_2 : pd.DataFrame
        Margins data (sums over the second variable)
    df_margins_3 : pd.DataFrame
        Margins data (sums over the third variable)
    var_names : list of strings
        Names of the variables over which we rake (e.g. cause, race, county)
    draws : string
        Names of the column containing the indices of the draws

    Returns
    -------
    sigma_ss : np.ndarray
        (I J + I K + J K - I - J - K + 1) * (I J + I K + J K - I - J - K + 1) covariance matrix
    """    
    nsamples = len(df_margins_1[draws].unique())
    var1 = df_margins_2[var_names[0]].unique().tolist()
    var2 = df_margins_1[var_names[1]].unique().tolist()
    var3 = df_margins_1[var_names[2]].unique().tolist()
    var1.sort()
    var2.sort()
    var3.sort()
    df1 = df_margins_1[[var_names[1], var_names[2], 'value_agg_over_' + var_names[0], draws]]
    df1 = df1.loc[(df1[var_names[1]].isin(var2[0:-1])) | ((df1[var_names[1]]==var2[-1]) & (df1[var_names[2]]==var3[-1]))]
    df1.sort_values(by=[var_names[2], var_names[1], draws], inplace=True)
    df2 = df_margins_2[[var_names[0], var_names[2], 'value_agg_over_' + var_names[1], draws]]
    df2 = df2.loc[df2[var_names[2]].isin(var3[0:-1])]
    df2.sort_values(by=[var_names[0], var_names[2], draws, inplace=True)
    df3 = df_margins_3[[var_names[0], var_names[1], 'value_agg_over_' + var_names[2], draws]]
    df3 = df3.loc[df3[var_names[0]].isin(var1[0:-1])]
    df3.sort_values(by=[var_names[1], var_names[0], draws, inplace=True)
    value1 = df1['value_agg_over_' + var_names[0]].to_numpy()
    value2 = df2['value_agg_over_' + var_names[1]].to_numpy()
    value3 = df3['value_agg_over_' + var_names[2]].to_numpy()
    value = np.concatenate((value1, value2, value3))
    X = np.reshape(value, shape=(nsamples, -1), order='F')
    Xmean = np.mean(X, axis=0)
    Xc = X - Xmean
    sigma_ss = np.matmul(np.transpose(Xc), Xc) / nsamples
    return sigma_ss

def compute_covariance_obs_margins_1D(
    df_obs: pd.DataFrame,
    df_margins: pd.DataFrame,
    var_names: list,
    draws: str
) -> np.ndarray:
    """Compute the covariance matrix of the observations and the margins in 1D.

    The observations will be sorted by var3, var2, var1, meaning that
    sigma_yy contains on its diagonal in this order the variances of
    y_111, ... , y_I11, y_121, ... , y_IJ1, y_112, ... , y_IJK.

    Parameters
    ----------
    df_obs : pd.DataFrame
        Observations data
    df_margins : pd.DataFrame
        Margins data (sums over the first variable)
    var_names : list of strings
        Names of the variables over which we rake (e.g. cause, race, county)
    draws : string
        Names of the column containing the indices of the draws

    Returns
    -------
    sigma_ys : np.ndarray
        (I * J * K) * 1 covariance matrix
    """
    nsamples = len(df_obs[draws].unique())
    var_names.reverse()
    df_obs = df_obs[['value'] + var_names + [draws]]
    df_obs.sort_values(by=var_names + [draws], inplace=True)
    df_margins = df_margins[['value_agg_over_' + var_names[0]] + [draws]]
    df_margins.sort_values(by=[draws], inplace=True)
    value_obs = df_obs['value'].to_numpy()
    X = np.reshape(value_obs, shape=(nsamples, -1), order='F')
    value_margins = df_margins['value_agg_over_' + var_names[0]].to_numpy()
    Y = np.reshape(value_margins, shape=(nsamples, -1), order='F')
    Xmean = np.mean(X, axis=0)
    Ymean = np.mean(Y, axis=0)
    Xc = X - Xmean
    Yc = Y - Ymean
    sigma_ys = np.matmul(np.transpose(Xc), Yc) / nsamples
    return sigma_ys

def compute_covariance_obs_margins_2D(
    df_obs: pd.DataFrame,
    df_margins_1: pd.DataFrame,
    df_margins_2: pd.DataFrame,
    var_names: list,
    draws: str
) -> np.ndarray:
    """Compute the covariance matrix of the observations and the margins in 2D.

    The observations will be sorted by var3, var2, var1, meaning that
    sigma_yy contains on its diagonal in this order the variances of
    y_111, ... , y_I11, y_121, ... , y_IJ1, y_112, ... , y_IJK.

    Parameters
    ----------
    df_obs : pd.DataFrame
        Observations data
    df_margins_1 : pd.DataFrame
        Margins data (sums over the first variable)
    df_margins_2 : pd.DataFrame
        Margins data (sums over the second variable)
    var_names : list of strings
        Names of the variables over which we rake (e.g. cause, race, county)
    draws : string
        Names of the column containing the indices of the draws

    Returns
    -------
    sigma_ys : np.ndarray
        (I * J * K) * (I + J - 1) covariance matrix
    """
    nsamples = len(df_obs[draws].unique())
    var_names.reverse()
    df_obs = df_obs[var_names + [draws]]
    df_obs.sort_values(by=var_names + [draws], inplace=True)
    df_margins_1 = df_margins_1[[var_names[1], 'value_agg_over_' + var_names[0], draws]]
    df_margins_1.sort_values(by=[var_names[1], draws], inplace=True)
    df_margins_2 = df_margins_2[[var_names[0], 'value_agg_over_' + var_names[1], draws]]
    df_margins_2.sort_values(by=[var_names[0], draws], inplace=True)
    value_obs = df_obs['value'].to_numpy()
    X = np.reshape(value_obs, shape=(nsamples, -1), order='F')
    value_margins_1 = df_margins_1['value_agg_over_' + var_names[0]].to_numpy()
    value_margins_2 = df_margins_2['value_agg_over_' + var_names[1]].to_numpy()
    value_margins = np.concatenate((value_margins_1, value_margins_2))
    Y = np.reshape(value_margins, shape=(nsamples, -1), order='F')
    Xmean = np.mean(X, axis=0)
    Ymean = np.mean(Y, axis=0)
    Xc = X - Xmean
    Yc = Y - Ymean
    sigma_ys = np.matmul(np.transpose(Xc), Yc) / nsamples
    return sigma_ys

def compute_covariance_obs_margins_3D(
    df_obs: pd.DataFrame,
    df_margins_1: pd.DataFrame,
    df_margins_2: pd.DataFrame,
    df_margins_3: pd.DataFrame,
    var_names: list,
    draws: str
) -> np.ndarray:
    """Compute the covariance matrix of the observations and the margins in 3D.

    The observations will be sorted by var3, var2, var1, meaning that
    sigma_yy contains on its diagonal in this order the variances of
    y_111, ... , y_I11, y_121, ... , y_IJ1, y_112, ... , y_IJK.

    Parameters
    ----------
    df_obs : pd.DataFrame
        Observations data
    df_margins_1 : pd.DataFrame
        Margins data (sums over the first variable)
    df_margins_2 : pd.DataFrame
        Margins data (sums over the second variable)
    df_margins_3 : pd.DataFrame
        Margins data (sums over the third variable)
    var_names : list of strings
        Names of the variables over which we rake (e.g. cause, race, county)
    draws : string
        Names of the column containing the indices of the draws

    Returns
    -------
    sigma_ys : np.ndarray
        (I * J * K) * (I J + I K + J K - I - J - K + 1) covariance matrix
    """
    nsamples = len(df_obs[draws].unique())
    var_names.reverse()
    df_obs = df_obs[var_names + [draws]]
    df_obs.sort_values(by=var_names + [draws], inplace=True)
    var1 = df_margins_2[var_names[0]].unique().tolist()
    var2 = df_margins_1[var_names[1]].unique().tolist()
    var3 = df_margins_1[var_names[2]].unique().tolist()
    var1.sort()
    var2.sort()
    var3.sort()
    df_margins_1 = df_margins_1[[var_names[1], var_names[2], 'value_agg_over_' + var_names[0], draws]]
    df_margins_1 = df_margins_1.loc[(df_margins_1[var_names[1]].isin(var2[0:-1])) | ((df_margins_1[var_names[1]]==var2[-1]) & (df_margins_1[var_names[2]]==var3[-1]))]
    df_margins_1.sort_values(by=[var_names[2], var_names[1], draws], inplace=True)
    df_margins_2 = df_margins_2[[var_names[0], var_names[2], 'value_agg_over_' + var_names[1], draws]]
    df_margins_2 = df_margins_2.loc[df_margins_2[var_names[2]].isin(var3[0:-1])]
    df_margins_2.sort_values(by=[var_names[0], var_names[2], draws, inplace=True)
    df_margins_3 = df_margins_3[[var_names[0], var_names[1], 'value_agg_over_' + var_names[2], draws]]
    df_margins_3 = df_margins_3.loc[df_margins_3[var_names[0]].isin(var1[0:-1])]
    df_margins_3.sort_values(by=[var_names[1], var_names[0], draws, inplace=True)
    value_obs = df_obs['value'].to_numpy()
    value_margins_1 = df_margins_1['value_agg_over_' + var_names[0]].to_numpy()
    value_margins_2 = df_margins_2['value_agg_over_' + var_names[1]].to_numpy()
    value_margins_3 = df_margins_3['value_agg_over_' + var_names[2]].to_numpy()
    value = np.concatenate((value_obs, value_margins_1, df_margins_2, df_margins_3))
    X = np.reshape(value, shape=(nsamples, -1), order='F')
    Xmean = np.mean(X, axis=0)
    Xc = X - Xmean
    sigma_ys = np.matmul(np.transpose(Xc), Xc) / nsamples
    return sigma_ys

def check_covariance(
    sigma_yy: np.ndarray,
    sigma_ss: np.ndarray,
    sigma_ys: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Check if the covariance matrix is definite positive.

    If it is not, assumes independence of the variables
    and return the diagonal matrix of the variances.

    Parameters
    ----------
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
    sigma = np.concatenate(( \
        np.concatenate((sigma_yy, sigma_ys), axis=1), \
        np.concatenate((np.transpose(sigma_ys), sigma_ss), axis=1)), axis=0)
    valid = True
    if np.allclose(np.transpose(sigma), sigma, rtol, atol):
        valid = False
    if np.any(np.linalg.eig(sigma)[0] < 0.0):
        valid = False
    if not valid:
        sigma_yy = np.diag(np.diag(sigma_yy))
        sigma_ss = np.diag(np.diag(sigma_ss))
        sigma_ys = np.zeros(sigma_ys.shape)
    return sigma_yy, sigma_ss, sigma_ys

