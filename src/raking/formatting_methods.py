"""Module with methods to format the data for the raking methods"""

import numpy as np
import pandas as pd

from pandas.api.types import CategoricalDtype

pd.options.mode.chained_assignment = None


def format_data_1D(
    df_obs: pd.DataFrame,
    df_margins: pd.DataFrame,
    var_name: str,
    weights: str = None,
    lower: str = None,
    upper: str = None,
) -> tuple[
    np.ndarray,
    float,
    int,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    """Read the data and create the inputs of the raking functions (1D problem).

    Parameters
    ----------
    df_obs : pd.DataFrame
        Observations data
    df_margins : pd.DataFrame
        Margins data
    var_name : string
        Name of the variable over which we rake (e.g. cause, race, county)
    weights : string
        Name of the column containing the raking weights
    lower : string
        Name of the column containing the lower boundaries (for logit raking)
    upper : string
        Name of the column containing the upper boundaries (for logit raking)

    Returns
    -------
    y : np.ndarray
        Vector of observations
    s : float
        Target sum of the observations s = sum_i y_i
    I : int
        Number of possible values for categorical variable 1
    q : np.ndarray
        Vector of weights
    l : np.ndarray
        Lower bounds for the observations
    h : np.ndarray
        Upper bounds for the observations
    """
    assert isinstance(df_obs, pd.DataFrame), (
        "The observations should be a pandas data frame."
    )
    assert len(df_obs) >= 2, (
        "There should be at least 2 data points for the observations."
    )

    assert isinstance(df_margins, pd.DataFrame), (
        "The margins should be a pandas data frame."
    )
    assert len(df_margins) == 1, (
        "There should be only one data point for the margins."
    )

    assert "value" in df_obs.columns.tolist(), (
        "The observations data frame should contain a value column."
    )
    assert isinstance(var_name, str), (
        "The name of the categorical variable should be a string."
    )
    assert var_name in df_obs.columns.tolist(), (
        "The column for the categorical variable "
        + var_name
        + " is missing from the observations data frame."
    )

    assert "value_agg_over_" + var_name in df_margins.columns.tolist(), (
        "The column for the aggregated value over "
        + var_name
        + " is missing from the margins data frame."
    )

    if weights is not None:
        assert isinstance(weights, str), (
            "The name of the column containing the weights should be a string."
        )
        assert weights in df_obs.columns.tolist(), (
            "The column containing the weights is missing from the data frame."
        )
    if lower is not None:
        assert isinstance(lower, str), (
            "The name of the column containing the lower boundaries should be a string."
        )
        assert lower in df_obs.columns.tolist(), (
            "The column containing the lower boundaries is missing from the data frame."
        )
    if upper is not None:
        assert isinstance(upper, str), (
            "The name of the column containing the upper boundaries should be a string."
        )
        assert upper in df_obs.columns.tolist(), (
            "The column containing the upper_boundaries is missing from the data frame."
        )

    # Check the observations data
    assert df_obs.value.isna().sum() == 0, (
        "There are missing values in the value column of the observations."
    )
    assert df_obs[var_name].isna().sum() == 0, (
        "There are missing values in the "
        + var_name
        + " column of the observations."
    )
    assert len(df_obs[df_obs.duplicated([var_name])]) == 0, (
        "There are duplicated rows in the observations."
    )

    # Check the margins data
    assert df_margins["value_agg_over_" + var_name].isna().sum() == 0, (
        "There are missing values in the value_agg_over"
        + var_name
        + " column of the margins."
    )

    # Create input variables for the raking functions
    df_obs.sort_values(by=[var_name], inplace=True)
    I = len(df_obs[var_name].unique())
    y = df_obs.value.to_numpy()
    s = df_margins["value_agg_over_" + var_name].to_numpy()[0]
    if weights is not None:
        q = df_obs[weights].to_numpy()
    else:
        q = None
    if lower is not None:
        l = df_obs[lower].to_numpy()
    else:
        l = None
    if upper is not None:
        h = df_obs[upper].to_numpy()
    else:
        h = None
    return (y, s, I, q, l, h)


def format_data_2D(
    df_obs: pd.DataFrame,
    df_margins_1: pd.DataFrame,
    df_margins_2: pd.DataFrame,
    var_names: list,
    weights: str = None,
    lower: str = None,
    upper: str = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    int,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    """Read the data and create the inputs of the raking functions (2D problem).

    Parameters
    ----------
    df_obs : pd.DataFrame
        Observations data
    df_margins_1 : pd.DataFrame
        Margins data (sums over the first variable)
    df_margins_2 : pd.DataFrame
        Margins data (sums over the second variable)
    var_names : list of 2 strings
        Names of the two variables over which we rake (e.g. cause, race, county)
    weights : string
        Name of the column containing the raking weights
    lower : string
        Name of the column containing the lower boundaries (for logit raking)
    upper : string
        Name of the column containing the upper boundaries (for logit raking)

    Returns
    -------
    y : np.ndarray
        Vector of observations
    s1 : np.ndarray
        Target sums of the observations over categorical variable 1
    s2 : np.ndarray
        Target sums of the observations over categorical variable 2
    I : int
        Number of possible values for categorical variable 1
    J : int
        Number of possible values for categorical variable 2
    q : np.ndarray
        Vector of weights
    l : np.ndarray
        Lower bounds for the observations
    h : np.ndarray
        Upper bounds for the observations
    """
    assert isinstance(df_obs, pd.DataFrame), (
        "The observations should be a pandas data frame."
    )
    assert len(df_obs) >= 4, (
        "There should be at least 4 data points for the observations."
    )

    assert isinstance(df_margins_1, pd.DataFrame), (
        "The margins for the first variable should be a pandas data frame."
    )
    assert len(df_margins_1) >= 2, (
        "There should be at least 2 data points for the first margins."
    )

    assert isinstance(df_margins_2, pd.DataFrame), (
        "The margins for the second variable should be a pandas data frame."
    )
    assert len(df_margins_2) >= 2, (
        "There should be at least 2 data points for the second margins."
    )

    assert "value" in df_obs.columns.tolist(), (
        "The observations data frame should contain a value column."
    )
    assert isinstance(var_names, list), (
        "Please enter the names of the columns containing the values of the categorical variables as a list."
    )
    assert len(var_names) == 2, "You should have 2 categorical variables."
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

    if weights is not None:
        assert isinstance(weights, str), (
            "The name of the column containing the weights should be a string."
        )
        assert weights in df_obs.columns.tolist(), (
            "The column containing the weights is missing from the data frame."
        )
    if lower is not None:
        assert isinstance(lower, str), (
            "The name of the column containing the lower boundaries should be a string."
        )
        assert lower in df_obs.columns.tolist(), (
            "The column containing the lower boundaries is missing from the data frame."
        )
    if upper is not None:
        assert isinstance(upper, str), (
            "The name of the column containing the upper boundaries should be a string."
        )
        assert upper in df_obs.columns.tolist(), (
            "The column containing the upper_boundaries is missing from the data frame."
        )

    # Check the observations data
    assert df_obs.value.isna().sum() == 0, (
        "There are missing values in the value column of the observations."
    )
    for var_name in var_names:
        assert df_obs[var_name].isna().sum() == 0, (
            "There are missing values in the "
            + var_name
            + " column of the observations."
        )
    assert len(df_obs[df_obs.duplicated(var_names)]) == 0, (
        "There are duplicated rows in the observations."
    )
    count_obs = df_obs[var_names].value_counts()
    assert (len(count_obs.unique()) == 1) and (count_obs.unique()[0] == 1), (
        "There are missing combinations of "
        + var_names[0]
        + " and "
        + var_names[1]
        + " in the observations."
    )

    # Check the first margins data
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
    assert len(df_margins_1[df_margins_1.duplicated([var_names[1]])]) == 0, (
        "There are duplicated rows in the first margins data frame."
    )

    # Check the second margins data
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
    assert len(df_margins_2[df_margins_2.duplicated([var_names[0]])]) == 0, (
        "There are duplicated rows in the second margins data frame."
    )

    # Check consistency between observations and margins
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

    # Create input variables for the raking functions
    df_obs.sort_values(by=[var_names[1], var_names[0]], inplace=True)
    df_margins_1.sort_values(by=[var_names[1]], inplace=True)
    df_margins_2.sort_values(by=[var_names[0]], inplace=True)
    I = len(df_obs[var_names[0]].unique())
    J = len(df_obs[var_names[1]].unique())
    y = df_obs.value.to_numpy()
    s1 = df_margins_1["value_agg_over_" + var_names[0]].to_numpy()
    s2 = df_margins_2["value_agg_over_" + var_names[1]].to_numpy()
    if weights is not None:
        q = df_obs[weights].to_numpy()
    else:
        q = None
    if lower is not None:
        l = df_obs[lower].to_numpy()
    else:
        l = None
    if upper is not None:
        h = df_obs[upper].to_numpy()
    else:
        h = None
    return (y, s1, s2, I, J, q, l, h)


def format_data_3D(
    df_obs: pd.DataFrame,
    df_margins_1: pd.DataFrame,
    df_margins_2: pd.DataFrame,
    df_margins_3: pd.DataFrame,
    var_names: list,
    weights: str = None,
    lower: str = None,
    upper: str = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    int,
    int,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    """Read the data and create the inputs of the raking functions (3D problem).

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
    var_names : list of 3 strings
        Names of the three variables over which we rake (e.g. cause, race, county)
    weights : string
        Name of the column containing the raking weights
    lower : string
        Name of the column containing the lower boundaries (for logit raking)
    upper : string
        Name of the column containing the upper boundaries (for logit raking)

    Returns
    -------
    y : np.ndarray
        Vector of observations
    s1 : np.ndarray
        Target sums of the observations over categorical variable 1
    s2 : np.ndarray
        Target sums of the observations over categorical variable 2
    s3 : np.ndarray
        Target sums of the observations over categorical variable 3
    I : int
        Number of possible values for categorical variable 1
    J : int
        Number of possible values for categorical variable 2
    K : int
        Number of possible values for categorical variable 3
    q : np.ndarray
        Vector of weights
    l : np.ndarray
        Lower bounds for the observations
    h : np.ndarray
        Upper bounds for the observations
    """
    assert isinstance(df_obs, pd.DataFrame), (
        "The observations should be a pandas data frame."
    )
    assert len(df_obs) >= 8, (
        "There should be at least 8 data points for the observations."
    )

    assert isinstance(df_margins_1, pd.DataFrame), (
        "The margins for the first variable should be a pandas data frame."
    )
    assert len(df_margins_1) >= 4, (
        "There should be at least 4 data points for the first margins."
    )

    assert isinstance(df_margins_2, pd.DataFrame), (
        "The margins for the second variable should be a pandas data frame."
    )
    assert len(df_margins_2) >= 4, (
        "There should be at least 4 data points for the second margins."
    )

    assert isinstance(df_margins_3, pd.DataFrame), (
        "The margins for the third variable should be a pandas data frame."
    )
    assert len(df_margins_3) >= 4, (
        "There should be at least 4 data points for the third margins."
    )

    assert "value" in df_obs.columns.tolist(), (
        "The observations data frame should contain a value column."
    )
    assert isinstance(var_names, list), (
        "Please enter the names of the columns containing the values of the categorical variables as a list."
    )
    assert len(var_names) == 3, "You should have 3 categorical variables."
    for var_name in var_names:
        assert isinstance(var_name, str), (
            "The name of the categorical variable "
            + +str(var_name)
            + " should be a string."
        )
        assert var_name in df_obs.columns.tolist(), (
            "The column for the categorical variable "
            + var_name
            + " is missing from the observations data frame."
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

    if weights is not None:
        assert isinstance(weights, str), (
            "The name of the column containing the weights should be a string."
        )
        assert weights in df_obs.columns.tolist(), (
            "The column containing the weights is missing from the data frame."
        )
    if lower is not None:
        assert isinstance(lower, str), (
            "The name of the column containing the lower boundaries should be a string."
        )
        assert lower in df_obs.columns.tolist(), (
            "The column containing the lower boundaries is missing from the data frame."
        )
    if upper is not None:
        assert isinstance(upper, str), (
            "The name of the column containing the upper boundaries should be a string."
        )
        assert upper in df_obs.columns.tolist(), (
            "The column containing the upper_boundaries is missing from the data frame."
        )

    # Check the observations data
    assert df_obs.value.isna().sum() == 0, (
        "There are missing values in the value column of the observations."
    )
    for var_name in var_names:
        assert df_obs[var_name].isna().sum() == 0, (
            "There are missing values in the "
            + var_name
            + " column of the observations."
        )
    assert len(df_obs[df_obs.duplicated(var_names)]) == 0, (
        "There are duplicated rows in the observations."
    )
    count_obs = df_obs[var_names].value_counts()
    assert (len(count_obs.unique()) == 1) and (count_obs.unique()[0] == 1), (
        "There are missing combinations of "
        + var_names[0]
        + ", "
        + var_names[1]
        + " and "
        + var_names[2]
        + " in the observations."
    )

    # Check the first margins data
    assert df_margins_1[var_names[1]].isna().sum() == 0, (
        "There are missing values in the "
        + var_names[1]
        + " column of the first margins."
    )
    assert df_margins_1[var_names[2]].isna().sum() == 0, (
        "There are missing values in the "
        + var_names[2]
        + " column of the first margins."
    )
    assert df_margins_1["value_agg_over_" + var_names[0]].isna().sum() == 0, (
        "There are missing values in the value_agg_over"
        + var_names[0]
        + " column of the first margins."
    )
    assert (
        len(df_margins_1[df_margins_1.duplicated([var_names[1], var_names[2]])])
        == 0
    ), "There are duplicated rows in the first margins data frame."
    count_obs = df_margins_1[[var_names[1], var_names[2]]].value_counts()
    assert (len(count_obs.unique()) == 1) and (count_obs.unique()[0] == 1), (
        "There are missing combinations of "
        + var_names[1]
        + " and "
        + var_names[2]
        + " in the first margins."
    )

    # Check the second margins data
    assert df_margins_2[var_names[0]].isna().sum() == 0, (
        "There are missing values in the "
        + var_names[0]
        + " column of the second margins."
    )
    assert df_margins_2[var_names[2]].isna().sum() == 0, (
        "There are missing values in the "
        + var_names[2]
        + " column of the second margins."
    )
    assert df_margins_2["value_agg_over_" + var_names[1]].isna().sum() == 0, (
        "There are missing values in the value_agg_over"
        + var_names[1]
        + " column of the second margins."
    )
    assert (
        len(df_margins_2[df_margins_2.duplicated([var_names[0], var_names[2]])])
        == 0
    ), "There are duplicated rows in the second margins data frame."
    count_obs = df_margins_2[[var_names[0], var_names[2]]].value_counts()
    assert (len(count_obs.unique()) == 1) and (count_obs.unique()[0] == 1), (
        "There are missing combinations of "
        + var_names[0]
        + " and "
        + var_names[2]
        + " in the second margins."
    )

    # Check the third margins data
    assert df_margins_3[var_names[0]].isna().sum() == 0, (
        "There are missing values in the "
        + var_names[0]
        + " column of the third margins."
    )
    assert df_margins_3[var_names[1]].isna().sum() == 0, (
        "There are missing values in the "
        + var_names[1]
        + " column of the third margins."
    )
    assert df_margins_3["value_agg_over_" + var_names[2]].isna().sum() == 0, (
        "There are missing values in the value_agg_over"
        + var_names[2]
        + " column of the third margins."
    )
    assert (
        len(df_margins_3[df_margins_3.duplicated([var_names[0], var_names[1]])])
        == 0
    ), "There are duplicated rows in the third margins data frame."
    count_obs = df_margins_3[[var_names[0], var_names[1]]].value_counts()
    assert (len(count_obs.unique()) == 1) and (count_obs.unique()[0] == 1), (
        "There are missing combinations of "
        + var_names[0]
        + " and "
        + var_names[1]
        + " in the third margins."
    )

    # Check consistency between observations and first margins
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

    # Check consistency between observations and second margins
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

    # Check consistency between observations and third margins
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

    # Create input variables for the raking functions
    df_obs.sort_values(
        by=[var_names[2], var_names[1], var_names[0]], inplace=True
    )
    df_margins_1.sort_values(by=[var_names[2], var_names[1]], inplace=True)
    df_margins_2.sort_values(by=[var_names[2], var_names[0]], inplace=True)
    df_margins_3.sort_values(by=[var_names[1], var_names[0]], inplace=True)
    I = len(df_obs[var_names[0]].unique())
    J = len(df_obs[var_names[1]].unique())
    K = len(df_obs[var_names[2]].unique())
    y = df_obs.value.to_numpy()
    s1 = (
        df_margins_1["value_agg_over_" + var_names[0]]
        .to_numpy()
        .reshape([J, K], order="F")
    )
    s2 = (
        df_margins_2["value_agg_over_" + var_names[1]]
        .to_numpy()
        .reshape([I, K], order="F")
    )
    s3 = (
        df_margins_3["value_agg_over_" + var_names[2]]
        .to_numpy()
        .reshape([I, J], order="F")
    )
    if weights is not None:
        q = df_obs[weights].to_numpy()
    else:
        q = None
    if lower is not None:
        l = df_obs[lower].to_numpy()
    else:
        l = None
    if upper is not None:
        h = df_obs[upper].to_numpy()
    else:
        h = None
    return (y, s1, s2, s3, I, J, K, q, l, h)


def format_data_USHD(
    df_obs: pd.DataFrame,
    df_margins: pd.DataFrame,
    margin_names: list,
    weights: str = None,
    lower: str = None,
    upper: str = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    int,
    int,
    int,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    """Read the data and create the inputs of the raking functions (USHD problem).

    Parameters
    ----------
    df_obs : pd.DataFrame
        Observations data
    df_margins : pd.DataFrame
        Margins data (GBD)
    margin_names : list
        Names for the all causes, all races, all counties categories (length 3)
    weights : string
        Name of the column containing the raking weights
    lower : string
        Name of the column containing the lower boundaries (for logit raking)
    upper : string
        Name of the column containing the upper boundaries (for logit raking)

    Returns
    -------
    y : np.ndarray
        Vector of observations
    s : np.ndarray
        Total number of deaths (all causes, and each cause)
    I : int
        Number of possible values for cause
    J : int
        Number of possible values for race
    K : int
        Number of possible values for county
    q : np.ndarray
        Vector of weights
    l : np.ndarray
        Lower bounds for the observations
    h : np.ndarray
        Upper bounds for the observations
    """
    assert isinstance(df_obs, pd.DataFrame), (
        "The observations should be a pandas data frame."
    )
    assert len(df_obs) >= 18, (
        "There should be at least 18 data points for the observations."
    )

    assert isinstance(df_margins, pd.DataFrame), (
        "The margins should be a pandas data frame."
    )
    assert len(df_margins) >= 3, (
        "There should be at least 3 data points for the margins."
    )

    for var_name in ["value", "cause", "race", "county"]:
        assert var_name in df_obs.columns.tolist(), (
            "The column for the categorical variable "
            + var_name
            + " is missing from the observations data frame."
        )

    assert "cause" in df_margins.columns.tolist(), (
        "The cause column is missing from the margins data frame."
    )
    assert "value_agg_over_race_county" in df_margins.columns.tolist(), (
        "The column for the aggregated value over races and counties is missing from the margins data frame."
    )

    assert isinstance(margin_names, list), (
        "Please enter the names of the all causes, all races, all counties categories as a list."
    )
    assert len(margin_names) == 3, (
        "There should be a margin name for each of the three variables cause, race, and county."
    )

    if weights is not None:
        assert isinstance(weights, str), (
            "The name of the column containing the weights should be a string."
        )
        assert weights in df_obs.columns.tolist(), (
            "The column containing the weights is missing from the data frame."
        )
    if lower is not None:
        assert isinstance(lower, str), (
            "The name of the column containing the lower boundaries should be a string."
        )
        assert lower in df_obs.columns.tolist(), (
            "The column containing the lower boundaries is missing from the data frame."
        )
    if upper is not None:
        assert isinstance(upper, str), (
            "The name of the column containing the upper boundaries should be a string."
        )
        assert upper in df_obs.columns.tolist(), (
            "The column containing the upper_boundaries is missing from the data frame."
        )

    # Check the observations data
    for var_name in ["value", "cause", "race", "county"]:
        assert df_obs[var_name].isna().sum() == 0, (
            "There are missing values in the "
            + var_name
            + " column of the observations."
        )
    assert len(df_obs[df_obs.duplicated(["cause", "race", "county"])]) == 0, (
        "There are duplicated rows in the observations."
    )
    count_obs = df_obs[["cause", "race", "county"]].value_counts()
    assert (len(count_obs.unique()) == 1) and (count_obs.unique()[0] == 1), (
        "There are missing combinations of cause, race and county in the observations."
    )

    # Check the margins data
    assert df_margins["cause"].isna().sum() == 0, (
        "There are missing values in the cause column of the margins."
    )
    assert df_margins["value_agg_over_race_county"].isna().sum() == 0, (
        "There are missing values in the value_agg_over_race_county column of the margins."
    )
    assert len(df_margins[df_margins.duplicated(["cause"])]) == 0, (
        "There are duplicated rows in the margins data frame."
    )

    # Check consistency between observations and margins
    assert len(df_obs["cause"].unique()) == len(df_margins["cause"].unique()), (
        "The number of categories for cause should be the same in the observations and margins data frames."
    )
    assert set(df_obs["cause"].unique().tolist()) == set(
        df_margins["cause"].unique().tolist()
    ), (
        "The names of the categories for cause should be the same in the observations and margins data frames."
    )

    # Create the sorting orders for the categorical variables
    cause_all = margin_names[0]
    race_all = margin_names[1]
    county_all = margin_names[2]
    cause_names = df_obs["cause"].unique().tolist()
    race_names = df_obs["race"].unique().tolist()
    county_names = df_obs["county"].unique().tolist()
    if cause_all in cause_names:
        cause_names.remove(cause_all)
    if race_all in race_names:
        race_names.remove(race_all)
    if county_all in county_names:
        county_names.remove(county_all)
    cause_ordering = [cause_all] + cause_names
    race_ordering = [race_all] + race_names
    county_ordering = [county_all] + county_names
    df_obs["cause"] = df_obs["cause"].astype(
        CategoricalDtype(categories=cause_ordering, ordered=True)
    )
    df_obs["race"] = df_obs["race"].astype(
        CategoricalDtype(categories=race_ordering, ordered=True)
    )
    df_obs["county"] = df_obs["county"].astype(
        CategoricalDtype(categories=county_ordering, ordered=True)
    )
    df_margins["cause"] = df_margins["cause"].astype(
        CategoricalDtype(categories=cause_ordering, ordered=True)
    )

    # Sort input observations and margins
    df_obs.sort_values(by=["county", "race", "cause"], inplace=True)
    df_margins.sort_values(by=["cause"], inplace=True)

    # Create input variables for the raking functions
    I = len(df_obs["cause"].unique()) - 1
    J = len(df_obs["race"].unique()) - 1
    K = len(df_obs["county"].unique())
    y = df_obs.value.to_numpy()
    s = df_margins["value_agg_over_race_county"].to_numpy()
    if weights is not None:
        q = df_obs[weights].to_numpy()
    else:
        q = None
    if lower is not None:
        l = df_obs[lower].to_numpy()
    else:
        l = None
    if upper is not None:
        h = df_obs[upper].to_numpy()
    else:
        h = None
    return (y, s, I, J, K, q, l, h)


def format_data_USHD_lower(
    df_obs: pd.DataFrame,
    df_margins_cause: pd.DataFrame,
    df_margins_county: pd.DataFrame,
    df_margins_all_causes: pd.DataFrame,
    margin_names: list,
    weights: str = None,
    lower: str = None,
    upper: str = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    int,
    int,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    """Read the data and create the inputs of the raking functions (USHD problem).

    Parameters
    ----------
    df_obs : pd.DataFrame
        Observations data
    df_margins_cause : pd.DataFrame
        Margins data (GBD: number of deaths from each cause, all races, all counties=state)
    df_margins_county: pd.DataFrame
        Margins data from previous level raking (Number of deaths for each county, all causes with same parent, all races)
    df_margins_all_causes: pd.DataFrame
        Margins data from previous level raking (Number of deaths for each race and each county, all causes with same parent)
    margin_names : list
        Names for the all causes, all races, all counties categories (length 3)
    weights : string
        Name of the column containing the raking weights
    lower : string
        Name of the column containing the lower boundaries (for logit raking)
    upper : string
        Name of the column containing the upper boundaries (for logit raking)

    Returns
    -------
    y : np.ndarray
        Vector of observations
    s_cause : np.ndarray
        Total number of deaths for each cause (all races, all counties)
    s_county: np.ndarray
        Number of deaths for each county (all causes, all races)
    s_all_causes: np.ndarray
        Number of deaths for each race and each county (all causes)
    I : int
        Number of possible values for cause
    J : int
        Number of possible values for race
    K : int
        Number of possible values for county
    q : np.ndarray
        Vector of weights
    l : np.ndarray
        Lower bounds for the observations
    h : np.ndarray
        Upper bounds for the observations
    """
    assert isinstance(df_obs, pd.DataFrame), (
        "The observations should be a pandas data frame."
    )
    assert len(df_obs) >= 12, (
        "There should be at least 12 data points for the observations."
    )

    assert isinstance(df_margins_cause, pd.DataFrame), (
        "The GBD margins should be a pandas data frame."
    )
    assert len(df_margins_cause) >= 2, (
        "There should be at least 2 data points for the GBD margins."
    )

    assert isinstance(df_margins_county, pd.DataFrame), (
        "The county margins should be a pandas data frame."
    )
    assert len(df_margins_county) >= 2, (
        "There should be at least 2 data points for the county margins."
    )

    assert isinstance(df_margins_all_causes, pd.DataFrame), (
        "The all causes margins should be a pandas data frame."
    )
    assert len(df_margins_all_causes) >= 4, (
        "There should be at least 4 data points for the all causes margins."
    )

    for var_name in ["value", "cause", "race", "county"]:
        assert var_name in df_obs.columns.tolist(), (
            "The column for the categorical variable "
            + var_name
            + " is missing from the observations data frame."
        )

    assert "cause" in df_margins_cause.columns.tolist(), (
        "The cause column is missing from the GBD margins data frame."
    )
    assert "value_agg_over_race_county" in df_margins_cause.columns.tolist(), (
        "The column for the aggregated value over races and counties is missing from the GBD margins data frame."
    )

    assert "county" in df_margins_county.columns.tolist(), (
        "The county column is missing from the county margins data frame."
    )
    assert "value_agg_over_cause_race" in df_margins_county.columns.tolist(), (
        "The column for the aggregated value over causes and races is missing from the county margins data frame."
    )

    for var_name in ["race", "county"]:
        assert var_name in df_margins_all_causes.columns.tolist(), (
            "The column for the categorical variable "
            + var_name
            + " is missing from the all causes margins data frame."
        )
    assert "value_agg_over_cause" in df_margins_all_causes.columns.tolist(), (
        "The column for the aggregated value over causes is missing from the all causes margins data frame."
    )

    assert isinstance(margin_names, list), (
        "Please enter the names of the all causes, all races, all counties categories as a list."
    )
    assert len(margin_names) == 3, (
        "There should be a margin name for each of the three variables cause, race, and county."
    )

    if weights is not None:
        assert isinstance(weights, str), (
            "The name of the column containing the weights should be a string."
        )
        assert weights in df_obs.columns.tolist(), (
            "The column containing the weights is missing from the data frame."
        )
    if lower is not None:
        assert isinstance(lower, str), (
            "The name of the column containing the lower boundaries should be a string."
        )
        assert lower in df_obs.columns.tolist(), (
            "The column containing the lower boundaries is missing from the data frame."
        )
    if upper is not None:
        assert isinstance(upper, str), (
            "The name of the column containing the upper boundaries should be a string."
        )
        assert upper in df_obs.columns.tolist(), (
            "The column containing the upper_boundaries is missing from the data frame."
        )

    # Check the observations data
    for var_name in ["value", "cause", "race", "county"]:
        assert df_obs[var_name].isna().sum() == 0, (
            "There are missing values in the "
            + var_name
            + " column of the observations."
        )
    assert len(df_obs[df_obs.duplicated(["cause", "race", "county"])]) == 0, (
        "There are duplicated rows in the observations."
    )
    count_obs = df_obs[["cause", "race", "county"]].value_counts()
    assert (len(count_obs.unique()) == 1) and (count_obs.unique()[0] == 1), (
        "There are missing combinations of cause, race and county in the observations."
    )

    # Check the GBD margins data
    assert df_margins_cause["cause"].isna().sum() == 0, (
        "There are missing values in the cause column of the GBD margins."
    )
    assert df_margins_cause["value_agg_over_race_county"].isna().sum() == 0, (
        "There are missing values in the value_agg_over_race_county column of the GBD margins."
    )
    assert len(df_margins_cause[df_margins_cause.duplicated(["cause"])]) == 0, (
        "There are duplicated rows in the GBD margins data frame."
    )

    # Check the county margins data
    assert df_margins_county["county"].isna().sum() == 0, (
        "There are missing values in the county column of the county margins."
    )
    assert df_margins_county["value_agg_over_cause_race"].isna().sum() == 0, (
        "There are missing values in the value_agg_over_cause_race column of the county margins."
    )
    assert (
        len(df_margins_county[df_margins_county.duplicated(["county"])]) == 0
    ), "There are duplicated rows in the county margins data frame."

    # Check the all causes margins data
    for var_name in ["race", "county"]:
        assert df_margins_all_causes[var_name].isna().sum() == 0, (
            "There are missing values in the "
            + var_name
            + " column of the all causes margins."
        )
    assert df_margins_all_causes["value_agg_over_cause"].isna().sum() == 0, (
        "There are missing values in the value_agg_over_cause column of the all causes margins."
    )
    assert (
        len(
            df_margins_all_causes[
                df_margins_all_causes.duplicated(["race", "county"])
            ]
        )
        == 0
    ), "There are duplicated rows in the all causes margins data frame."
    count_obs = df_margins_all_causes[["race", "county"]].value_counts()
    assert (len(count_obs.unique()) == 1) and (count_obs.unique()[0] == 1), (
        "There are missing combinations of race and county in the all causes margins."
    )

    # Check consistency between observations and GBD margins
    assert len(df_obs["cause"].unique()) == len(
        df_margins_cause["cause"].unique()
    ), (
        "The number of categories for cause should be the same in the observations and GBD margins data frames."
    )
    assert set(df_obs["cause"].unique().tolist()) == set(
        df_margins_cause["cause"].unique().tolist()
    ), (
        "The names of the categories for cause should be the same in the observations and GBD margins data frames."
    )

    # Check consistency between observations and county margins
    assert len(df_obs["county"].unique()) == len(
        df_margins_county["county"].unique()
    ), (
        "The number of categories for county should be the same in the observations and county margins data frames."
    )
    assert set(df_obs["county"].unique().tolist()) == set(
        df_margins_county["county"].unique().tolist()
    ), (
        "The names of the categories for county should be the same in the observations and county margins data frames."
    )

    # Check consistency between observations and all causes margins
    assert len(df_obs["race"].unique()) - 1 == len(
        df_margins_all_causes["race"].unique()
    ), (
        "The number of categories for race should be the same in the observations and all causes margins data frames."
    )
    assert len(df_obs["county"].unique()) == len(
        df_margins_all_causes["county"].unique()
    ), (
        "The number of categories for county should be the same in the observations and all causes margins data frames."
    )
    set_race_obs = set(df_obs["race"].unique().tolist())
    set_race_obs.remove(margin_names[1])
    assert set_race_obs == set(
        df_margins_all_causes["race"].unique().tolist()
    ), (
        "The names of the categories for race should be the same in the observations and all_causes margins data frames."
    )
    assert set(df_obs["county"].unique().tolist()) == set(
        df_margins_all_causes["county"].unique().tolist()
    ), (
        "The names of the categories for county should be the same in the observations and all_causes margins data frames."
    )

    # Create the sorting orders for the categorical variables
    cause_all = margin_names[0]
    race_all = margin_names[1]
    county_all = margin_names[2]
    cause_names = df_obs["cause"].unique().tolist()
    race_names = df_obs["race"].unique().tolist()
    county_names = df_obs["county"].unique().tolist()
    if cause_all in cause_names:
        cause_names.remove(cause_all)
    if race_all in race_names:
        race_names.remove(race_all)
    if county_all in county_names:
        county_names.remove(county_all)
    cause_ordering = [cause_all] + cause_names
    race_ordering = [race_all] + race_names
    county_ordering = [county_all] + county_names
    df_obs["cause"] = df_obs["cause"].astype(
        CategoricalDtype(categories=cause_ordering, ordered=True)
    )
    df_obs["race"] = df_obs["race"].astype(
        CategoricalDtype(categories=race_ordering, ordered=True)
    )
    df_obs["county"] = df_obs["county"].astype(
        CategoricalDtype(categories=county_ordering, ordered=True)
    )
    df_margins_cause["cause"] = df_margins_cause["cause"].astype(
        CategoricalDtype(categories=cause_ordering, ordered=True)
    )
    df_margins_county["county"] = df_margins_county["county"].astype(
        CategoricalDtype(categories=county_ordering, ordered=True)
    )
    df_margins_all_causes["race"] = df_margins_all_causes["race"].astype(
        CategoricalDtype(categories=race_ordering, ordered=True)
    )
    df_margins_all_causes["county"] = df_margins_all_causes["county"].astype(
        CategoricalDtype(categories=county_ordering, ordered=True)
    )

    # Sort input observations and margins
    df_obs.sort_values(by=["county", "race", "cause"], inplace=True)
    df_margins_cause.sort_values(by=["cause"], inplace=True)
    df_margins_county.sort_values(by=["county"], inplace=True)
    df_margins_all_causes.sort_values(by=["county", "race"], inplace=True)

    # Create input variables for the raking functions
    I = len(df_obs["cause"].unique())
    J = len(df_obs["race"].unique()) - 1
    K = len(df_obs["county"].unique())
    y = df_obs.value.to_numpy()
    s_cause = df_margins_cause["value_agg_over_race_county"].to_numpy()
    s_county = df_margins_county["value_agg_over_cause_race"].to_numpy()
    s_all_causes = (
        df_margins_all_causes["value_agg_over_cause"]
        .to_numpy()
        .reshape((J, K), order="F")
    )
    if weights is not None:
        q = df_obs[weights].to_numpy()
    else:
        q = None
    if lower is not None:
        l = df_obs[lower].to_numpy()
    else:
        l = None
    if upper is not None:
        h = df_obs[upper].to_numpy()
    else:
        h = None
    return (y, s_cause, s_county, s_all_causes, I, J, K, q, l, h)
