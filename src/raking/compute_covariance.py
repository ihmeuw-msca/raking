"""Module with methods to compute the covaraince matrices of observations and margins"""

import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None


def check_observations(
    df_obs: pd.DataFrame, var_names: list, draws: str
) -> None:
    """Check whether the observations data frame is valid.

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
    None
    """
    assert isinstance(
        df_obs, pd.DataFrame
    ), "The observations should be a pandas data frame."
    assert (
        len(df_obs) >= 2
    ), "There should be at least 2 data points for the observations."

    assert (
        "value" in df_obs.columns.tolist()
    ), "The observations data frame should contain a value column."

    assert isinstance(
        var_names, list
    ), "Please enter the names of the columns containing the values of the categorical variables as a list."
    for var_name in var_names:
        assert isinstance(var_name, str), (
            "The name of the categorical variable "
            + str(var_name)
            + " should be a string."
        )
        assert var_name in df_obs.columns.tolist(), (
            "The column for the categorical variable "
            + var_name
            + " is missing from the observations data frame."
        )

    assert isinstance(
        draws, str
    ), "The name of the column containing the draws should be a string."
    assert (
        draws in df_obs.columns.tolist()
    ), "The column containing the draws is missing from the observations data frame."

    assert (
        df_obs.value.isna().sum() == 0
    ), "There are missing values in the value column of the observations."
    for var_name in var_names:
        assert df_obs[var_name].isna().sum() == 0, (
            "There are missing values in the "
            + var_name
            + " column of the observations."
        )
    assert (
        df_obs[draws].isna().sum() == 0
    ), "There are missing values in the draws column of the observations."
    assert (
        len(df_obs[df_obs.duplicated(var_names + [draws])]) == 0
    ), "There are duplicated rows in the observations."
    count_obs = df_obs[var_names + [draws]].value_counts()
    assert (
        (len(count_obs.unique()) == 1) and (count_obs.unique()[0] == 1)
    ), "There are missing combinations of variables and draws in the observations."


def compute_covariance_obs(
    df_obs: pd.DataFrame, var_names: list, draws: str
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
    check_observations(df_obs, var_names, draws)

    nsamples = len(df_obs[draws].unique())
    var_names_reverse = var_names.copy()
    var_names_reverse.reverse()
    df = df_obs[["value"] + var_names + [draws]]
    df.sort_values(by=var_names_reverse + [draws], inplace=True)
    value = df["value"].to_numpy()
    X = np.reshape(value, shape=(nsamples, -1), order="F")
    Xmean = np.mean(X, axis=0)
    Xc = X - Xmean
    sigma_yy = np.matmul(np.transpose(Xc), Xc) / nsamples
    return sigma_yy


def check_margins_1D(
    df_margins: pd.DataFrame, var_names: list, draws: str
) -> None:
    """Check whether the margin data frame is valid in 1D.

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
    None
    """
    assert isinstance(
        df_margins, pd.DataFrame
    ), "The margins should be a pandas data frame."
    assert (
        len(df_margins) >= 1
    ), "There should be at least 1 data point for the margins."

    assert isinstance(
        var_names, list
    ), "Please enter the names of the columns containing the values of the categorical variables as a list."
    assert len(var_names) == 1, "You should have 1 categorical variable."
    assert isinstance(
        var_names[0], str
    ), "The name of the categorical variable should be a string."
    assert "value_agg_over_" + var_names[0] in df_margins.columns.tolist(), (
        "The column for the aggregated value over "
        + var_names[0]
        + " is missing from the margins data frame."
    )

    assert isinstance(
        draws, str
    ), "The name of the column containing the draws should be a string."
    assert (
        draws in df_margins.columns.tolist()
    ), "The column containing the draws is missing from the margins data frame."

    assert df_margins["value_agg_over_" + var_names[0]].isna().sum() == 0, (
        "There are missing values in the value_agg_over"
        + var_names[0]
        + " column of the margins."
    )
    assert (
        df_margins[draws].isna().sum() == 0
    ), "There are missing values in the draws column of the margins."
    assert (
        len(df_margins[df_margins.duplicated([draws])]) == 0
    ), "There are duplicated rows in the margins."
    count_obs = df_margins[[draws]].value_counts()
    assert (len(count_obs.unique()) == 1) and (
        count_obs.unique()[0] == 1
    ), "There are missing draws in the margins."


def compute_covariance_margins_1D(
    df_margins: pd.DataFrame, var_names: list, draws: str
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
    check_margins_1D(df_margins, var_names, draws)

    nsamples = len(df_margins[draws].unique())
    df = df_margins[["value_agg_over_" + var_names[0]] + [draws]]
    df.sort_values(by=[draws], inplace=True)
    value = df["value_agg_over_" + var_names[0]].to_numpy()
    X = np.reshape(value, shape=(nsamples, -1), order="F")
    Xmean = np.mean(X, axis=0)
    Xc = X - Xmean
    sigma_ss = np.matmul(np.transpose(Xc), Xc) / nsamples
    return sigma_ss


def check_margins_2D(
    df_margins_1: pd.DataFrame,
    df_margins_2: pd.DataFrame,
    var_names: list,
    draws: str,
) -> None:
    """Check whether the margins data frame are valid in 2D.

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
    None
    """
    assert isinstance(
        df_margins_1, pd.DataFrame
    ), "The margins for the first variable should be a pandas data frame."
    assert (
        len(df_margins_1) >= 2
    ), "There should be at least 2 data points for the first margins."

    assert isinstance(
        df_margins_2, pd.DataFrame
    ), "The margins for the second variable should be a pandas data frame."
    assert (
        len(df_margins_2) >= 2
    ), "There should be at least 2 data points for the second margins."

    assert isinstance(
        var_names, list
    ), "Please enter the names of the columns containing the values of the categorical variables as a list."
    assert len(var_names) == 2, "You should have 2 categorical variables."
    for var_name in var_names:
        assert isinstance(var_name, str), (
            "The name of the categorical variable "
            + str(var_name)
            + " should be a string."
        )

    assert var_names[1] in df_margins_1.columns.tolist(), (
        "The column for the categorigal variable "
        + var_name[1]
        + " is missing from the first margins data frame."
    )
    assert "value_agg_over_" + var_names[0] in df_margins_1.columns.tolist(), (
        "The column for the aggregated value over "
        + var_names[0]
        + " is missing from the first margins data frame."
    )

    assert var_names[0] in df_margins_2.columns.tolist(), (
        "The column for the categorigal variable "
        + var_name[0]
        + " is missing from the second margins data frame."
    )
    assert "value_agg_over_" + var_names[1] in df_margins_2.columns.tolist(), (
        "The column for the aggregated value over "
        + var_names[1]
        + " is missing from the second margins data frame."
    )

    assert isinstance(
        draws, str
    ), "The name of the column containing the draws should be a string."
    assert (
        draws in df_margins_1.columns.tolist()
    ), "The column containing the draws is missing from the first margins data frame."
    assert (
        draws in df_margins_2.columns.tolist()
    ), "The column containing the draws is missing from the second margins data frame."

    assert df_margins_1[var_names[1]].isna().sum() == 0, (
        "There are missing values in the "
        + var_names[1]
        + " column of the margins."
    )
    assert df_margins_1["value_agg_over_" + var_names[0]].isna().sum() == 0, (
        "There are missing values in the value_agg_over"
        + var_names[0]
        + " column of the margins."
    )
    assert (
        df_margins_1[draws].isna().sum() == 0
    ), "There are missing values in the draws column of the first margins."
    assert (
        len(df_margins_1[df_margins_1.duplicated([var_names[1], draws])]) == 0
    ), "There are duplicated rows in the first margins data frame."
    count_obs = df_margins_1[[var_names[1], draws]].value_counts()
    assert (len(count_obs.unique()) == 1) and (count_obs.unique()[0] == 1), (
        "There are missing combinations of "
        + var_names[1]
        + " and draws in the first margins."
    )

    assert df_margins_2[var_names[0]].isna().sum() == 0, (
        "There are missing values in the "
        + var_names[0]
        + " column of the margins."
    )
    assert df_margins_2["value_agg_over_" + var_names[1]].isna().sum() == 0, (
        "There are missing values in the value_agg_over"
        + var_names[1]
        + " column of the margins."
    )
    assert (
        df_margins_2[draws].isna().sum() == 0
    ), "There are missing values in the draws column of the second margins."
    assert (
        len(df_margins_2[df_margins_2.duplicated([var_names[0], draws])]) == 0
    ), "There are duplicated rows in the second margins data frame."
    count_obs = df_margins_2[[var_names[0], draws]].value_counts()
    assert (len(count_obs.unique()) == 1) and (count_obs.unique()[0] == 1), (
        "There are missing combinations of "
        + var_names[0]
        + " and draws in the second margins."
    )


def compute_covariance_margins_2D(
    df_margins_1: pd.DataFrame,
    df_margins_2: pd.DataFrame,
    var_names: list,
    draws: str,
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
    check_margins_2D(df_margins_1, df_margins_2, var_names, draws)

    nsamples = len(df_margins_1[draws].unique())
    df1 = df_margins_1[[var_names[1], "value_agg_over_" + var_names[0], draws]]
    df1.sort_values(by=[var_names[1], draws], inplace=True)
    df2 = df_margins_2[[var_names[0], "value_agg_over_" + var_names[1], draws]]
    df2.sort_values(by=[var_names[0], draws], inplace=True)
    value1 = df1["value_agg_over_" + var_names[0]].to_numpy()
    value2 = df2["value_agg_over_" + var_names[1]].to_numpy()
    value = np.concatenate((value1, value2))
    X = np.reshape(value, shape=(nsamples, -1), order="F")
    X = X[:, 0:-1]
    Xmean = np.mean(X, axis=0)
    Xc = X - Xmean
    sigma_ss = np.matmul(np.transpose(Xc), Xc) / nsamples
    return sigma_ss


def check_margins_3D(
    df_margins_1: pd.DataFrame,
    df_margins_2: pd.DataFrame,
    df_margins_3: pd.DataFrame,
    var_names: list,
    draws: str,
) -> None:
    """Check whether the margins data frames are valid in 3D.

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
    None
    """
    assert isinstance(
        df_margins_1, pd.DataFrame
    ), "The margins for the first variable should be a pandas data frame."
    assert (
        len(df_margins_1) >= 4
    ), "There should be at least 4 data points for the first margins."

    assert isinstance(
        df_margins_2, pd.DataFrame
    ), "The margins for the second variable should be a pandas data frame."
    assert (
        len(df_margins_2) >= 4
    ), "There should be at least 4 data points for the second margins."

    assert isinstance(
        df_margins_3, pd.DataFrame
    ), "The margins for the second variable should be a pandas data frame."
    assert (
        len(df_margins_3) >= 4
    ), "There should be at least 4 data points for the third margins."

    assert isinstance(
        var_names, list
    ), "Please enter the names of the columns containing the values of the categorical variables as a list."
    assert len(var_names) == 3, "You should have 3 categorical variables."
    for var_name in var_names:
        assert isinstance(var_name, str), (
            "The name of the categorical variable "
            + str(var_name)
            + " should be a string."
        )

    assert var_names[1] in df_margins_1.columns.tolist(), (
        "The column for the categorigal variable "
        + var_name[1]
        + " is missing from the first margins data frame."
    )
    assert var_names[2] in df_margins_1.columns.tolist(), (
        "The column for the categorigal variable "
        + var_name[2]
        + " is missing from the first margins data frame."
    )
    assert "value_agg_over_" + var_names[0] in df_margins_1.columns.tolist(), (
        "The column for the aggregated value over "
        + var_names[0]
        + " is missing from the first margins data frame."
    )

    assert var_names[0] in df_margins_2.columns.tolist(), (
        "The column for the categorigal variable "
        + var_name[0]
        + " is missing from the second margins data frame."
    )
    assert var_names[2] in df_margins_2.columns.tolist(), (
        "The column for the categorigal variable "
        + var_name[2]
        + " is missing from the second margins data frame."
    )
    assert "value_agg_over_" + var_names[1] in df_margins_2.columns.tolist(), (
        "The column for the aggregated value over "
        + var_names[1]
        + " is missing from the second margins data frame."
    )

    assert var_names[0] in df_margins_3.columns.tolist(), (
        "The column for the categorigal variable "
        + var_name[0]
        + " is missing from the third margins data frame."
    )
    assert var_names[1] in df_margins_3.columns.tolist(), (
        "The column for the categorigal variable "
        + var_name[1]
        + " is missing from the third margins data frame."
    )
    assert "value_agg_over_" + var_names[2] in df_margins_3.columns.tolist(), (
        "The column for the aggregated value over "
        + var_names[2]
        + " is missing from the third margins data frame."
    )

    assert isinstance(
        draws, str
    ), "The name of the column containing the draws should be a string."
    assert (
        draws in df_margins_1.columns.tolist()
    ), "The column containing the draws is missing from the first margins data frame."
    assert (
        draws in df_margins_2.columns.tolist()
    ), "The column containing the draws is missing from the second margins data frame."
    assert (
        draws in df_margins_3.columns.tolist()
    ), "The column containing the draws is missing from the third margins data frame."

    assert df_margins_1[var_names[1]].isna().sum() == 0, (
        "There are missing values in the "
        + var_names[1]
        + " column of the margins."
    )
    assert df_margins_1[var_names[2]].isna().sum() == 0, (
        "There are missing values in the "
        + var_names[2]
        + " column of the margins."
    )
    assert df_margins_1["value_agg_over_" + var_names[0]].isna().sum() == 0, (
        "There are missing values in the value_agg_over"
        + var_names[0]
        + " column of the margins."
    )
    assert (
        df_margins_1[draws].isna().sum() == 0
    ), "There are missing values in the draws column of the first margins."
    assert (
        len(
            df_margins_1[
                df_margins_1.duplicated([var_names[1], var_names[2], draws])
            ]
        )
        == 0
    ), "There are duplicated rows in the first margins data frame."
    count_obs = df_margins_1[[var_names[1], var_names[2], draws]].value_counts()
    assert (len(count_obs.unique()) == 1) and (count_obs.unique()[0] == 1), (
        "There are missing combinations of "
        + var_names[1]
        + ", "
        + var_names[2]
        + " and draws in the first margins."
    )

    assert df_margins_2[var_names[0]].isna().sum() == 0, (
        "There are missing values in the "
        + var_names[0]
        + " column of the margins."
    )
    assert df_margins_2[var_names[2]].isna().sum() == 0, (
        "There are missing values in the "
        + var_names[2]
        + " column of the margins."
    )
    assert df_margins_2["value_agg_over_" + var_names[1]].isna().sum() == 0, (
        "There are missing values in the value_agg_over"
        + var_names[1]
        + " column of the margins."
    )
    assert (
        df_margins_2[draws].isna().sum() == 0
    ), "There are missing values in the draws column of the second margins."
    assert (
        len(
            df_margins_2[
                df_margins_2.duplicated([var_names[0], var_names[2], draws])
            ]
        )
        == 0
    ), "There are duplicated rows in the second margins data frame."
    count_obs = df_margins_2[[var_names[0], var_names[2], draws]].value_counts()
    assert (len(count_obs.unique()) == 1) and (count_obs.unique()[0] == 1), (
        "There are missing combinations of "
        + var_names[0]
        + ", "
        + var_names[2]
        + " and draws in the second margins."
    )

    assert df_margins_3[var_names[0]].isna().sum() == 0, (
        "There are missing values in the "
        + var_names[0]
        + " column of the margins."
    )
    assert df_margins_3[var_names[1]].isna().sum() == 0, (
        "There are missing values in the "
        + var_names[1]
        + " column of the margins."
    )
    assert df_margins_3["value_agg_over_" + var_names[2]].isna().sum() == 0, (
        "There are missing values in the value_agg_over"
        + var_names[2]
        + " column of the margins."
    )
    assert (
        df_margins_3[draws].isna().sum() == 0
    ), "There are missing values in the draws column of the third margins."
    assert (
        len(
            df_margins_3[
                df_margins_3.duplicated([var_names[0], var_names[1], draws])
            ]
        )
        == 0
    ), "There are duplicated rows in the third margins data frame."
    count_obs = df_margins_3[[var_names[0], var_names[1], draws]].value_counts()
    assert (len(count_obs.unique()) == 1) and (count_obs.unique()[0] == 1), (
        "There are missing combinations of "
        + var_names[0]
        + ", "
        + var_names[1]
        + " and draws in the third margins."
    )


def compute_covariance_margins_3D(
    df_margins_1: pd.DataFrame,
    df_margins_2: pd.DataFrame,
    df_margins_3: pd.DataFrame,
    var_names: list,
    draws: str,
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
    check_margins_3D(df_margins_1, df_margins_2, df_margins_3, var_names, draws)

    nsamples = len(df_margins_1[draws].unique())
    var1 = df_margins_2[var_names[0]].unique().tolist()
    var2 = df_margins_1[var_names[1]].unique().tolist()
    var3 = df_margins_1[var_names[2]].unique().tolist()
    var1.sort()
    var2.sort()
    var3.sort()
    df1 = df_margins_1[
        [var_names[1], var_names[2], "value_agg_over_" + var_names[0], draws]
    ]
    df1 = df1.loc[
        (df1[var_names[1]].isin(var2[0:-1]))
        | ((df1[var_names[1]] == var2[-1]) & (df1[var_names[2]] == var3[-1]))
    ]
    df1.sort_values(by=[var_names[2], var_names[1], draws], inplace=True)
    df2 = df_margins_2[
        [var_names[0], var_names[2], "value_agg_over_" + var_names[1], draws]
    ]
    df2 = df2.loc[df2[var_names[2]].isin(var3[0:-1])]
    df2.sort_values(by=[var_names[0], var_names[2], draws], inplace=True)
    df3 = df_margins_3[
        [var_names[0], var_names[1], "value_agg_over_" + var_names[2], draws]
    ]
    df3 = df3.loc[df3[var_names[0]].isin(var1[0:-1])]
    df3.sort_values(by=[var_names[1], var_names[0], draws], inplace=True)
    value1 = df1["value_agg_over_" + var_names[0]].to_numpy()
    value2 = df2["value_agg_over_" + var_names[1]].to_numpy()
    value3 = df3["value_agg_over_" + var_names[2]].to_numpy()
    value = np.concatenate((value1, value2, value3))
    X = np.reshape(value, shape=(nsamples, -1), order="F")
    Xmean = np.mean(X, axis=0)
    Xc = X - Xmean
    sigma_ss = np.matmul(np.transpose(Xc), Xc) / nsamples
    return sigma_ss


def check_obs_margins_1D(
    df_obs: pd.DataFrame, df_margins: pd.DataFrame, draws: str
) -> None:
    """Check whether the observations and margins data frames are consistent in 1D.

    Parameters
    ----------
    df_obs : pd.DataFrame
        Observations data
    df_margins : pd.DataFrame
        Margins data (sums over the first variable)
    draws : string
        Names of the column containing the indices of the draws

    Returns
    -------
    None
    """
    assert (
        set(df_obs[draws].unique().tolist())
        == set(df_margins[draws].unique().tolist())
    ), "The draws should be the same in the observations and margins data frames."


def compute_covariance_obs_margins_1D(
    df_obs: pd.DataFrame, df_margins: pd.DataFrame, var_names: list, draws: str
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
    check_observations(df_obs, var_names, draws)
    check_margins_1D(df_margins, var_names, draws)
    check_obs_margins_1D(df_obs, df_margins, draws)

    nsamples = len(df_obs[draws].unique())
    var_names_reverse = var_names.copy()
    var_names_reverse.reverse()
    df_obs = df_obs[["value"] + var_names + [draws]]
    df_obs.sort_values(by=var_names_reverse + [draws], inplace=True)
    df_margins = df_margins[["value_agg_over_" + var_names[0]] + [draws]]
    df_margins.sort_values(by=[draws], inplace=True)
    value_obs = df_obs["value"].to_numpy()
    X = np.reshape(value_obs, shape=(nsamples, -1), order="F")
    value_margins = df_margins["value_agg_over_" + var_names[0]].to_numpy()
    Y = np.reshape(value_margins, shape=(nsamples, -1), order="F")
    Xmean = np.mean(X, axis=0)
    Ymean = np.mean(Y, axis=0)
    Xc = X - Xmean
    Yc = Y - Ymean
    sigma_ys = np.matmul(np.transpose(Xc), Yc) / nsamples
    return sigma_ys


def check_obs_margins_2D(
    df_obs: pd.DataFrame,
    df_margins_1: pd.DataFrame,
    df_margins_2: pd.DataFrame,
    var_names: list,
    draws: str,
) -> None:
    """Check whether the observations and margins data frames are consistent in 2D.

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
    None
    """
    assert (
        set(df_obs[draws].unique().tolist())
        == set(df_margins_1[draws].unique().tolist())
    ), "The draws should be the same in the observations and the first margins data frames."

    assert (
        set(df_obs[draws].unique().tolist())
        == set(df_margins_2[draws].unique().tolist())
    ), "The draws should be the same in the observations and the second margins data frames."

    assert len(df_obs[var_names[0]].unique()) == len(
        df_margins_2[var_names[0]].unique()
    ), (
        "The number of categories for "
        + var_names[0]
        + " should be the same in the observations and margins data frames."
    )
    assert set(df_obs[var_names[0]].unique().tolist()) == set(
        df_margins_2[var_names[0]].unique().tolist()
    ), (
        "The names of the categories for "
        + var_names[0]
        + " should be the same in the observations and margins data frames."
    )
    assert len(df_obs[var_names[1]].unique()) == len(
        df_margins_1[var_names[1]].unique()
    ), (
        "The number of categories for "
        + var_names[1]
        + " should be the same in the observations and margins data frames."
    )
    assert set(df_obs[var_names[1]].unique().tolist()) == set(
        df_margins_1[var_names[1]].unique().tolist()
    ), (
        "The names of the categories for "
        + var_names[1]
        + " should be the same in the observations and margins data frames."
    )


def compute_covariance_obs_margins_2D(
    df_obs: pd.DataFrame,
    df_margins_1: pd.DataFrame,
    df_margins_2: pd.DataFrame,
    var_names: list,
    draws: str,
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
    check_observations(df_obs, var_names, draws)
    check_margins_2D(df_margins_1, df_margins_2, var_names, draws)
    check_obs_margins_2D(df_obs, df_margins_1, df_margins_2, var_names, draws)

    nsamples = len(df_obs[draws].unique())
    var_names_reverse = var_names.copy()
    var_names_reverse.reverse()
    df_obs = df_obs[["value"] + var_names + [draws]]
    df_obs.sort_values(by=var_names_reverse + [draws], inplace=True)
    df_margins_1 = df_margins_1[
        [var_names[1], "value_agg_over_" + var_names[0], draws]
    ]
    df_margins_1.sort_values(by=[var_names[1], draws], inplace=True)
    df_margins_2 = df_margins_2[
        [var_names[0], "value_agg_over_" + var_names[1], draws]
    ]
    df_margins_2.sort_values(by=[var_names[0], draws], inplace=True)
    value_obs = df_obs["value"].to_numpy()
    X = np.reshape(value_obs, shape=(nsamples, -1), order="F")
    value_margins_1 = df_margins_1["value_agg_over_" + var_names[0]].to_numpy()
    value_margins_2 = df_margins_2["value_agg_over_" + var_names[1]].to_numpy()
    value_margins = np.concatenate((value_margins_1, value_margins_2))
    Y = np.reshape(value_margins, shape=(nsamples, -1), order="F")
    Y = Y[:, 0:-1]
    Xmean = np.mean(X, axis=0)
    Ymean = np.mean(Y, axis=0)
    Xc = X - Xmean
    Yc = Y - Ymean
    sigma_ys = np.matmul(np.transpose(Xc), Yc) / nsamples
    return sigma_ys


def check_obs_margins_3D(
    df_obs: pd.DataFrame,
    df_margins_1: pd.DataFrame,
    df_margins_2: pd.dataFrame,
    df_margins_3: pd.dataFrame,
    var_names: list,
    draws: str,
) -> None:
    """Check whether the observations and margins data frame are consistent in 3D.

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
    None
    """
    assert (
        set(df_obs[draws].unique().tolist())
        == set(df_margins_1[draws].unique().tolist())
    ), "The draws should be the same in the observations and the first margins data frames."

    assert (
        set(df_obs[draws].unique().tolist())
        == set(df_margins_2[draws].unique().tolist())
    ), "The draws should be the same in the observations and the second margins data frames."

    assert (
        set(df_obs[draws].unique().tolist())
        == set(df_margins_3[draws].unique().tolist())
    ), "The draws should be the same in the observations and the third margins data frames."

    assert (
        set(df_obs[draws].unique().tolist())
        == set(df_margins_2[draws].unique().tolist())
    ), "The draws should be the same in the observations and the second margins data frames."

    assert len(df_obs[var_names[1]].unique()) == len(
        df_margins_1[var_names[1]].unique()
    ), (
        "The number of categories for "
        + var_names[1]
        + " should be the same in the observations and first margins data frames."
    )
    assert set(df_obs[var_names[1]].unique().tolist()) == set(
        df_margins_1[var_names[1]].unique().tolist()
    ), (
        "The names of the categories for "
        + var_names[1]
        + " should be the same in the observations and first margins data frames."
    )
    assert len(df_obs[var_names[2]].unique()) == len(
        df_margins_1[var_names[2]].unique()
    ), (
        "The number of categories for "
        + var_names[2]
        + " should be the same in the observations and first margins data frames."
    )
    assert set(df_obs[var_names[2]].unique().tolist()) == set(
        df_margins_1[var_names[2]].unique().tolist()
    ), (
        "The names of the categories for "
        + var_names[2]
        + " should be the same in the observations and first margins data frames."
    )

    assert len(df_obs[var_names[0]].unique()) == len(
        df_margins_2[var_names[0]].unique()
    ), (
        "The number of categories for "
        + var_names[0]
        + " should be the same in the observations and second margins data frames."
    )
    assert set(df_obs[var_names[0]].unique().tolist()) == set(
        df_margins_2[var_names[0]].unique().tolist()
    ), (
        "The names of the categories for "
        + var_names[0]
        + " should be the same in the observations and second margins data frames."
    )
    assert len(df_obs[var_names[2]].unique()) == len(
        df_margins_2[var_names[2]].unique()
    ), (
        "The number of categories for "
        + var_names[2]
        + " should be the same in the observations and second margins data frames."
    )
    assert set(df_obs[var_names[2]].unique().tolist()) == set(
        df_margins_2[var_names[2]].unique().tolist()
    ), (
        "The names of the categories for "
        + var_names[2]
        + " should be the same in the observations and second margins data frames."
    )

    assert len(df_obs[var_names[0]].unique()) == len(
        df_margins_3[var_names[0]].unique()
    ), (
        "The number of categories for "
        + var_names[0]
        + " should be the same in the observations and third margins data frames."
    )
    assert set(df_obs[var_names[0]].unique().tolist()) == set(
        df_margins_3[var_names[0]].unique().tolist()
    ), (
        "The names of the categories for "
        + var_names[0]
        + " should be the same in the observations and third margins data frames."
    )
    assert len(df_obs[var_names[1]].unique()) == len(
        df_margins_3[var_names[1]].unique()
    ), (
        "The number of categories for "
        + var_names[1]
        + " should be the same in the observations and third margins data frames."
    )
    assert set(df_obs[var_names[1]].unique().tolist()) == set(
        df_margins_3[var_names[1]].unique().tolist()
    ), (
        "The names of the categories for "
        + var_names[1]
        + " should be the same in the observations and third margins data frames."
    )


def compute_covariance_obs_margins_3D(
    df_obs: pd.DataFrame,
    df_margins_1: pd.DataFrame,
    df_margins_2: pd.DataFrame,
    df_margins_3: pd.DataFrame,
    var_names: list,
    draws: str,
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
    check_observations(df_obs, var_names, draws)
    check_margins_3D(df_margins_1, df_margins_2, df_margins_3, var_names, draws)
    check_obs_margins_3D(
        df_obs, df_margins_1, df_margins_2, df_margins_3, var_names, draws
    )

    nsamples = len(df_obs[draws].unique())
    var_names_reverse = var_names.copy()
    var_names_reverse.reverse()
    df_obs = df_obs[["value"] + var_names + [draws]]
    df_obs.sort_values(by=var_names_reverse + [draws], inplace=True)
    var1 = df_margins_2[var_names[0]].unique().tolist()
    var2 = df_margins_1[var_names[1]].unique().tolist()
    var3 = df_margins_1[var_names[2]].unique().tolist()
    var1.sort()
    var2.sort()
    var3.sort()
    df_margins_1 = df_margins_1[
        [var_names[1], var_names[2], "value_agg_over_" + var_names[0], draws]
    ]
    df_margins_1 = df_margins_1.loc[
        (df_margins_1[var_names[1]].isin(var2[0:-1]))
        | (
            (df_margins_1[var_names[1]] == var2[-1])
            & (df_margins_1[var_names[2]] == var3[-1])
        )
    ]
    df_margins_1.sort_values(
        by=[var_names[2], var_names[1], draws], inplace=True
    )
    df_margins_2 = df_margins_2[
        [var_names[0], var_names[2], "value_agg_over_" + var_names[1], draws]
    ]
    df_margins_2 = df_margins_2.loc[df_margins_2[var_names[2]].isin(var3[0:-1])]
    df_margins_2.sort_values(
        by=[var_names[0], var_names[2], draws], inplace=True
    )
    df_margins_3 = df_margins_3[
        [var_names[0], var_names[1], "value_agg_over_" + var_names[2], draws]
    ]
    df_margins_3 = df_margins_3.loc[df_margins_3[var_names[0]].isin(var1[0:-1])]
    df_margins_3.sort_values(
        by=[var_names[1], var_names[0], draws], inplace=True
    )
    value_obs = df_obs["value"].to_numpy()
    X = np.reshape(value_obs, shape=(nsamples, -1), order="F")
    value_margins_1 = df_margins_1["value_agg_over_" + var_names[0]].to_numpy()
    value_margins_2 = df_margins_2["value_agg_over_" + var_names[1]].to_numpy()
    value_margins_3 = df_margins_3["value_agg_over_" + var_names[2]].to_numpy()
    value_margins = np.concatenate(
        (value_margins_1, value_margins_2, value_margins_3)
    )
    Y = np.reshape(value_margins, shape=(nsamples, -1), order="F")
    Xmean = np.mean(X, axis=0)
    Ymean = np.mean(Y, axis=0)
    Xc = X - Xmean
    Yc = Y - Ymean
    sigma_ys = np.matmul(np.transpose(Xc), Yc) / nsamples
    return sigma_ys


def check_format_covariance(
    sigma_yy: np.ndarray, sigma_ss: np.ndarray, sigma_ys: np.ndarray
) -> None:
    """Check the format of the input covariance matrices.

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
    None
    """
    assert isinstance(
        sigma_yy, np.ndarray
    ), "The covariance matrix of the observations should be a Numpy array."
    assert (
        len(sigma_yy.shape) == 2
    ), "The covariance matrix of the observations should be a 2D Numpy array."
    assert (
        np.shape(sigma_yy)[0] == np.shape(sigma_yy)[1]
    ), "The covariance matrix of the observations should be a square matrix."
    assert isinstance(
        sigma_ss, np.ndarray
    ), "The covariance matrix of the margins should be a Numpy array."
    assert (
        len(sigma_ss.shape) == 2
    ), "The covariance matrix of the margins should be a 2D Numpy array."
    assert (
        np.shape(sigma_ss)[0] == np.shape(sigma_ss)[1]
    ), "The covariance matrix of the margins should be a square matrix."
    assert isinstance(
        sigma_ys, np.ndarray
    ), "The covariance matrix of the observations and margins should be a Numpy array."
    assert (
        len(sigma_ys.shape) == 2
    ), "The covariance matrix of the observations and margins should be a 2D Numpy array."
    assert (
        np.shape(sigma_ys)[0] == np.shape(sigma_yy)[0]
    ), "The covariance matrix of observations and margins should have the same number of rows as the covariance matrix of the observations."
    assert (
        np.shape(sigma_ys)[1] == np.shape(sigma_ss)[1]
    ), "The covariance matrix of observations and margins should have the same number of columns as the covariance matrix of the margins."


def check_covariance(
    sigma_yy: np.ndarray,
    sigma_ss: np.ndarray,
    sigma_ys: np.ndarray,
    rtol: float = 1e-05,
    atol: float = 1e-08,
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
    rtol : float
        Relative tolerance to check whether the margins are consistant. See numpy.allclose documentation for details.
    atol : float
        Absolute tolerance to check whether the margins are consistant. See numpy.allclose documentation for details.

    Returns
    -------
    sigma_yy : np.ndarray
        Covariance matrix of the observations
    sigma_ss : np.ndarray
        Covariance matrix of the margins
    sigma_ys : np.ndarray
        Covariance matrix of the observations and margins
    """
    check_format_covariance(sigma_yy, sigma_ss, sigma_ys)
    sigma = np.concatenate(
        (
            np.concatenate((sigma_yy, sigma_ys), axis=1),
            np.concatenate((np.transpose(sigma_ys), sigma_ss), axis=1),
        ),
        axis=0,
    )
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
