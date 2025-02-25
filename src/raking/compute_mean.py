"""Module with methods to compute the mean of observations and margins"""

import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None


def compute_mean(
    df_obs: pd.DataFrame,
    df_margins: list,
    var_names: list = None,
    draws: str = "draws",
    use_case: str = None,
) -> tuple[pd.DataFrame, list]:
    """Compute the means of the values over all the samples.

    Parameters
    ----------
    df_obs : pd.DataFrame
        Observations data
    df_margins : list of pd.DataFrame
        list of data frames contatining the margins data
    var_names : list of strings
        Names of the variables over which we rake (e.g. cause, race, county).
        None if using special case.
    draws: string
        Name of the column that contains the samples.
    use_case : string
        Name of the use case if using special case.

    Returns
    -------
    df_obs_mean : pd.DataFrame
        Means of observations data
    df_margins_mean : list of pd.DataFrame
        list of data frames contatining the mans of the margins data
    """
    assert isinstance(draws, str), (
        "The name of the column containing the draws must be a string."
    )
    if use_case == None:
        assert isinstance(var_names, list), (
            "The variables over which we rake must be entered as a list."
        )
        assert len(var_names) in [1, 2, 3], (
            "The dimension of the raking problem must be 1, 2 or 3"
        )
        dim = len(var_names)
    else:
        assert use_case in ["USHD"], (
            "Only the special use case 'USHD' is currently implemented."
        )
        dim = "USHD"
        var_names = ["cause"]
    columns = df_obs.columns.drop([draws, "value"]).to_list()
    df_obs_mean = (
        df_obs.groupby(columns).mean().reset_index().drop(columns=[draws])
    )
    df_margins_mean = []
    for df_margin, var_name in zip(df_margins, var_names):
        if use_case == "USHD":
            value_name = "value_agg_over_race_county"
        else:
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
