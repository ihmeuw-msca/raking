import pytest
import numpy as np
import pandas as pd
from raking.run_raking import RakingData
from raking.compute_mean import compute_mean
from raking.compute_covariance import compute_covariance


def test_run_raking_1D_covariance(example_1D_draws):
    df_obs = example_1D_draws.df_obs
    df_margin = example_1D_draws.df_margin
    (df_obs_mean, df_margins_mean) = compute_mean(
        df_obs, [df_margin], ["var1"], "draws"
    )
    (sigma_yy, sigma_ss, sigma_ys) = compute_covariance(
        df_obs, [df_margin], ["var1"], "draws"
    )
    data = RakingData(
        df_obs=df_obs_mean,
        df_margins=df_margins_mean,
        var_names=["var1"],
    )
    data.rake()
    data.compute_variance(
        sigma_yy=sigma_yy, sigma_ss=sigma_ss, sigma_ys=sigma_ys
    )
    assert np.allclose(
        data.df_obs["raked_value"].sum(),
        df_margins_mean[0]["value_agg_over_var1"].iloc[0],
    ), "The raked values do not sum to the margin."


def test_run_raking_2D_covariance(example_2D_draws):
    df_obs = example_2D_draws.df_obs
    df_margins = [example_2D_draws.df_margins_1, example_2D_draws.df_margins_2]
    (df_obs_mean, df_margins_mean) = compute_mean(
        df_obs, df_margins, ["var1", "var2"], "draws"
    )
    (sigma_yy, sigma_ss, sigma_ys) = compute_covariance(
        df_obs, df_margins, ["var1", "var2"], "draws"
    )
    data = RakingData(
        df_obs=df_obs_mean,
        df_margins=df_margins_mean,
        var_names=["var1", "var2"],
    )
    data.rake()
    data.compute_variance(
        sigma_yy=sigma_yy, sigma_ss=sigma_ss, sigma_ys=sigma_ys
    )
    sum_over_var1 = (
        data.df_obs.groupby(["var2"])
        .agg({"raked_value": "sum"})
        .reset_index()
        .merge(df_margins_mean[0], on="var2")
    )
    assert np.allclose(
        sum_over_var1["raked_value"], sum_over_var1["value_agg_over_var1"]
    ), "The sums over the first variable must match the first margins."
    sum_over_var2 = (
        data.df_obs.groupby(["var1"])
        .agg({"raked_value": "sum"})
        .reset_index()
        .merge(df_margins_mean[1], on="var1")
    )
    assert np.allclose(
        sum_over_var2["raked_value"], sum_over_var2["value_agg_over_var2"]
    ), "The sums over the second variable must match the second margins."


def test_run_raking_3D_covariance(example_3D_draws):
    df_obs = example_3D_draws.df_obs
    df_margins = [
        example_3D_draws.df_margins_1,
        example_3D_draws.df_margins_2,
        example_3D_draws.df_margins_3,
    ]
    (df_obs_mean, df_margins_mean) = compute_mean(
        df_obs, df_margins, ["var1", "var2", "var3"], "draws"
    )
    (sigma_yy, sigma_ss, sigma_ys) = compute_covariance(
        df_obs, df_margins, ["var1", "var2", "var3"], "draws"
    )
    data = RakingData(
        df_obs=df_obs_mean,
        df_margins=df_margins_mean,
        var_names=["var1", "var2", "var3"],
    )
    data.rake()
    data.compute_variance(
        sigma_yy=sigma_yy, sigma_ss=sigma_ss, sigma_ys=sigma_ys
    )
    sum_over_var1 = (
        data.df_obs.groupby(["var2", "var3"])
        .agg({"raked_value": "sum"})
        .reset_index()
        .merge(df_margins_mean[0], on=["var2", "var3"])
    )
    assert np.allclose(
        sum_over_var1["raked_value"], sum_over_var1["value_agg_over_var1"]
    ), "The sums over the first variable must match the first margins."
    sum_over_var2 = (
        data.df_obs.groupby(["var1", "var3"])
        .agg({"raked_value": "sum"})
        .reset_index()
        .merge(df_margins_mean[1], on=["var1", "var3"])
    )
    assert np.allclose(
        sum_over_var2["raked_value"], sum_over_var2["value_agg_over_var2"]
    ), "The sums over the second variable must match the second margins."
    sum_over_var3 = (
        data.df_obs.groupby(["var1", "var2"])
        .agg({"raked_value": "sum"})
        .reset_index()
        .merge(df_margins_mean[2], on=["var1", "var2"])
    )
    assert np.allclose(
        sum_over_var3["raked_value"], sum_over_var3["value_agg_over_var3"]
    ), "The sums over the third variable must match the third margins."


def test_run_raking_USHD_covariance(example_USHD_draws):
    df_obs = example_USHD_draws.df_obs
    df_margins = [example_USHD_draws.df_margins]
    (df_obs_mean, df_margins_mean) = compute_mean(
        df_obs, df_margins, use_case="USHD", draws="draws"
    )
    (sigma_yy, sigma_ss, sigma_ys) = compute_covariance(
        df_obs, df_margins, use_case="USHD", draws="draws"
    )
    data = RakingData(
        df_obs=df_obs_mean, df_margins=df_margins_mean, use_case="USHD"
    )
    data.rake()
    data.compute_variance(
        sigma_yy=sigma_yy, sigma_ss=sigma_ss, sigma_ys=sigma_ys
    )
    sum_over_cause = (
        data.df_obs.loc[data.df_obs.cause != "_all"]
        .groupby(["race", "county"])
        .agg({"raked_value": "sum"})
        .reset_index()
        .merge(
            data.df_obs.loc[data.df_obs.cause == "_all"], on=["race", "county"]
        )
    )
    assert np.allclose(
        sum_over_cause["raked_value_x"],
        sum_over_cause["raked_value_y"],
        atol=1.0e-4,
    ), "The sums over the cause must match the all causes deaths."
    sum_over_race = (
        data.df_obs.loc[data.df_obs.race != 0]
        .groupby(["cause", "county"])
        .agg({"raked_value": "sum"})
        .reset_index()
        .merge(data.df_obs.loc[data.df_obs.race == 0], on=["cause", "county"])
    )
    assert np.allclose(
        sum_over_race["raked_value_x"],
        sum_over_race["raked_value_y"],
        atol=1.0e-4,
    ), "The sums over the race must match the all races deaths."
    sum_over_race_county = (
        data.df_obs.loc[data.df_obs.race != 0]
        .groupby(["cause"])
        .agg({"raked_value": "sum"})
        .reset_index()
        .merge(df_margins_mean[0], on=["cause"])
    )
    assert np.allclose(
        sum_over_race_county["raked_value"],
        sum_over_race_county["value_agg_over_race_county"],
        atol=1.0e-5,
    ), "The sums over race and county must match the GBD values."
