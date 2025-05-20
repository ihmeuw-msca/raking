import pytest
import numpy as np
from raking.run_raking import run_raking


def test_run_raking_1D(example_1D):
    (df_obs, Dphi_y, Dphi_s, sigma) = run_raking(
        dim=1,
        df_obs=example_1D.df_obs,
        df_margins=[example_1D.df_margin],
        var_names=["var1"],
        cov_mat=False,
    )
    assert np.allclose(
        df_obs["raked_value"].sum(),
        example_1D.df_margin["value_agg_over_var1"].iloc[0],
    ), "The raked values do not sum to the margin."


def test_run_raking_2D(example_2D):
    (df_obs, Dphi_y, Dphi_s, sigma) = run_raking(
        dim=2,
        df_obs=example_2D.df_obs,
        df_margins=[example_2D.df_margins_1, example_2D.df_margins_2],
        var_names=["var1", "var2"],
        cov_mat=False,
    )
    sum_over_var1 = (
        df_obs.groupby(["var2"])
        .agg({"raked_value": "sum"})
        .reset_index()
        .merge(example_2D.df_margins_1, on="var2")
    )
    assert np.allclose(
        sum_over_var1["raked_value"], sum_over_var1["value_agg_over_var1"]
    ), "The sums over the first variable must match the first margins."
    sum_over_var2 = (
        df_obs.groupby(["var1"])
        .agg({"raked_value": "sum"})
        .reset_index()
        .merge(example_2D.df_margins_2, on="var1")
    )
    assert np.allclose(
        sum_over_var2["raked_value"], sum_over_var2["value_agg_over_var2"]
    ), "The sums over the second variable must match the second margins."


def test_run_raking_3D(example_3D):
    (df_obs, Dphi_y, Dphi_s, sigma) = run_raking(
        dim=3,
        df_obs=example_3D.df_obs,
        df_margins=[
            example_3D.df_margins_1,
            example_3D.df_margins_2,
            example_3D.df_margins_3,
        ],
        var_names=["var1", "var2", "var3"],
        cov_mat=False,
    )
    sum_over_var1 = (
        df_obs.groupby(["var2", "var3"])
        .agg({"raked_value": "sum"})
        .reset_index()
        .merge(example_3D.df_margins_1, on=["var2", "var3"])
    )
    assert np.allclose(
        sum_over_var1["raked_value"], sum_over_var1["value_agg_over_var1"]
    ), "The sums over the first variable must match the first margins."
    sum_over_var2 = (
        df_obs.groupby(["var1", "var3"])
        .agg({"raked_value": "sum"})
        .reset_index()
        .merge(example_3D.df_margins_2, on=["var1", "var3"])
    )
    assert np.allclose(
        sum_over_var2["raked_value"], sum_over_var2["value_agg_over_var2"]
    ), "The sums over the second variable must match the second margins."
    sum_over_var3 = (
        df_obs.groupby(["var1", "var2"])
        .agg({"raked_value": "sum"})
        .reset_index()
        .merge(example_3D.df_margins_3, on=["var1", "var2"])
    )
    assert np.allclose(
        sum_over_var3["raked_value"], sum_over_var3["value_agg_over_var3"]
    ), "The sums over the third variable must match the third margins."


def test_run_raking_USHD(example_USHD):
    (df_obs, Dphi_y, Dphi_s, sigma) = run_raking(
        dim="USHD",
        df_obs=example_USHD.df_obs,
        df_margins=[example_USHD.df_margins],
        var_names=None,
        margin_names=["_all"],
        cov_mat=False,
    )
    sum_over_cause = (
        df_obs.loc[df_obs.cause != "_all"]
        .groupby(["race", "county"])
        .agg({"raked_value": "sum"})
        .reset_index()
        .merge(df_obs.loc[df_obs.cause == "_all"], on=["race", "county"])
    )
    assert np.allclose(
        sum_over_cause["raked_value_x"],
        sum_over_cause["raked_value_y"],
        atol=1.0e-4,
    ), "The sums over the cause must match the all causes deaths."
    sum_over_race = (
        df_obs.loc[df_obs.race != 0]
        .groupby(["cause", "county"])
        .agg({"raked_value": "sum"})
        .reset_index()
        .merge(df_obs.loc[df_obs.race == 0], on=["cause", "county"])
    )
    assert np.allclose(
        sum_over_race["raked_value_x"],
        sum_over_race["raked_value_y"],
        atol=1.0e-4,
    ), "The sums over the race must match the all races deaths."
    sum_over_race_county = (
        df_obs.loc[df_obs.race != 0]
        .groupby(["cause"])
        .agg({"raked_value": "sum"})
        .reset_index()
        .merge(example_USHD.df_margins, on=["cause"])
    )
    assert np.allclose(
        sum_over_race_county["raked_value"],
        sum_over_race_county["value_agg_over_race_county"],
        atol=1.0e-5,
    ), "The sums over race and county must match the GBD values."


def test_run_raking_1D_draws(example_1D_draws):
    (df_obs, Dphi_y, Dphi_s, sigma) = run_raking(
        dim=1,
        df_obs=example_1D_draws.df_obs,
        df_margins=[example_1D_draws.df_margin],
        var_names=["var1"],
        draws="draws",
        cov_mat=True,
    )
    assert np.allclose(
        df_obs["raked_value"].sum(),
        example_1D_draws.df_margin["value_agg_over_var1"].mean(),
    ), "The raked values do not sum to the margin."


def test_run_raking_2D_draws(example_2D_draws):
    (df_obs, Dphi_y, Dphi_s, sigma) = run_raking(
        dim=2,
        df_obs=example_2D_draws.df_obs,
        df_margins=[
            example_2D_draws.df_margins_1,
            example_2D_draws.df_margins_2,
        ],
        var_names=["var1", "var2"],
        draws="draws",
        cov_mat=True,
    )
    sum_over_var1 = (
        df_obs.groupby(["var2"])
        .agg({"raked_value": "sum"})
        .reset_index()
        .merge(
            example_2D_draws.df_margins_1.groupby(["var2"])
            .agg({"value_agg_over_var1": "mean"})
            .reset_index(),
            on="var2",
        )
    )
    assert np.allclose(
        sum_over_var1["raked_value"], sum_over_var1["value_agg_over_var1"]
    ), "The sums over the first variable must match the first margins."
    sum_over_var2 = (
        df_obs.groupby(["var1"])
        .agg({"raked_value": "sum"})
        .reset_index()
        .merge(
            example_2D_draws.df_margins_2.groupby(["var1"])
            .agg({"value_agg_over_var2": "mean"})
            .reset_index(),
            on="var1",
        )
    )
    assert np.allclose(
        sum_over_var2["raked_value"], sum_over_var2["value_agg_over_var2"]
    ), "The sums over the second variable must match the second margins."


def test_run_raking_3D_draws(example_3D_draws):
    (df_obs, Dphi_y, Dphi_s, sigma) = run_raking(
        dim=3,
        df_obs=example_3D_draws.df_obs,
        df_margins=[
            example_3D_draws.df_margins_1,
            example_3D_draws.df_margins_2,
            example_3D_draws.df_margins_3,
        ],
        var_names=["var1", "var2", "var3"],
        draws="draws",
        cov_mat=True,
    )
    sum_over_var1 = (
        df_obs.groupby(["var2", "var3"])
        .agg({"raked_value": "sum"})
        .reset_index()
        .merge(
            example_3D_draws.df_margins_1.groupby(["var2", "var3"])
            .agg({"value_agg_over_var1": "mean"})
            .reset_index(),
            on=["var2", "var3"],
        )
    )
    assert np.allclose(
        sum_over_var1["raked_value"], sum_over_var1["value_agg_over_var1"]
    ), "The sums over the first variable must match the first margins."
    sum_over_var2 = (
        df_obs.groupby(["var1", "var3"])
        .agg({"raked_value": "sum"})
        .reset_index()
        .merge(
            example_3D_draws.df_margins_2.groupby(["var1", "var3"])
            .agg({"value_agg_over_var2": "mean"})
            .reset_index(),
            on=["var1", "var3"],
        )
    )
    assert np.allclose(
        sum_over_var2["raked_value"], sum_over_var2["value_agg_over_var2"]
    ), "The sums over the second variable must match the second margins."
    sum_over_var3 = (
        df_obs.groupby(["var1", "var2"])
        .agg({"raked_value": "sum"})
        .reset_index()
        .merge(
            example_3D_draws.df_margins_3.groupby(["var1", "var2"])
            .agg({"value_agg_over_var3": "mean"})
            .reset_index(),
            on=["var1", "var2"],
        )
    )
    assert np.allclose(
        sum_over_var3["raked_value"], sum_over_var3["value_agg_over_var3"]
    ), "The sums over the third variable must match the third margins."


def test_run_raking_USHD_draws(example_USHD_draws):
    (df_obs, Dphi_y, Dphi_s, sigma) = run_raking(
        dim="USHD",
        df_obs=example_USHD_draws.df_obs,
        df_margins=[example_USHD_draws.df_margins],
        var_names=None,
        margin_names=["_all"],
        cov_mat=True,
    )
    sum_over_cause = (
        df_obs.loc[df_obs.cause != "_all"]
        .groupby(["race", "county"])
        .agg({"raked_value": "sum"})
        .reset_index()
        .merge(df_obs.loc[df_obs.cause == "_all"], on=["race", "county"])
    )
    assert np.allclose(
        sum_over_cause["raked_value_x"],
        sum_over_cause["raked_value_y"],
        atol=1.0e-4,
    ), "The sums over the cause must match the all causes deaths."
    sum_over_race = (
        df_obs.loc[df_obs.race != 0]
        .groupby(["cause", "county"])
        .agg({"raked_value": "sum"})
        .reset_index()
        .merge(df_obs.loc[df_obs.race == 0], on=["cause", "county"])
    )
    assert np.allclose(
        sum_over_race["raked_value_x"],
        sum_over_race["raked_value_y"],
        atol=1.0e-4,
    ), "The sums over the race must match the all races deaths."
    sum_over_race_county = (
        df_obs.loc[df_obs.race != 0]
        .groupby(["cause"])
        .agg({"raked_value": "sum"})
        .reset_index()
        .merge(
            example_USHD_draws.df_margins.groupby(["cause"])
            .agg({"value_agg_over_race_county": "mean"})
            .reset_index(),
            on=["cause"],
        )
    )
    assert np.allclose(
        sum_over_race_county["raked_value"],
        sum_over_race_county["value_agg_over_race_county"],
        atol=1.0e-5,
    ), "The sums over race and county must match the GBD values."


def test_run_raking_1D_weights(example_1D_draws):
    df_obs = example_1D_draws.df_obs
    df_obs = (
        df_obs.groupby(["var1"]).agg({"value": ["mean", "std"]}).reset_index()
    )
    df_obs.columns = [" ".join(col).strip() for col in df_obs.columns.values]
    df_obs.rename(
        columns={"value mean": "value", "value std": "weight"}, inplace=True
    )
    df_obs["lower"] = 1.8
    df_obs["upper"] = 3.2
    df_margin = example_1D_draws.df_margin
    df_margin = df_margin[["value_agg_over_var1"]].mean().to_frame().transpose()
    (df_obs, Dphi_y, Dphi_s, sigma) = run_raking(
        dim=1,
        df_obs=df_obs,
        df_margins=[df_margin],
        var_names=["var1"],
        cov_mat=False,
        method="logit",
        weights="weight",
        lower="lower",
        upper="upper",
    )
    assert np.allclose(
        df_obs["raked_value"].sum(),
        df_margin["value_agg_over_var1"].iloc[0],
    ), "The raked values do not sum to the margin."


def test_run_raking_2D_weights(example_2D_draws):
    df_obs = example_2D_draws.df_obs
    df_obs = (
        df_obs.groupby(["var1", "var2"])
        .agg({"value": ["mean", "std"]})
        .reset_index()
    )
    df_obs.columns = [" ".join(col).strip() for col in df_obs.columns.values]
    df_obs.rename(
        columns={"value mean": "value", "value std": "weight"}, inplace=True
    )
    df_obs["lower"] = 1.8
    df_obs["upper"] = 3.2
    df_margins_1 = example_2D_draws.df_margins_1
    df_margins_1 = (
        df_margins_1.groupby(["var2"])
        .agg({"value_agg_over_var1": "mean"})
        .reset_index()
    )
    df_margins_2 = example_2D_draws.df_margins_2
    df_margins_2 = (
        df_margins_2.groupby(["var1"])
        .agg({"value_agg_over_var2": "mean"})
        .reset_index()
    )
    (df_obs, Dphi_y, Dphi_s, sigma) = run_raking(
        dim=2,
        df_obs=df_obs,
        df_margins=[
            df_margins_1,
            df_margins_2,
        ],
        var_names=["var1", "var2"],
        cov_mat=False,
        method="logit",
        weights="weight",
        lower="lower",
        upper="upper",
    )
    sum_over_var1 = (
        df_obs.groupby(["var2"])
        .agg({"raked_value": "sum"})
        .reset_index()
        .merge(df_margins_1, on="var2")
    )
    assert np.allclose(
        sum_over_var1["raked_value"], sum_over_var1["value_agg_over_var1"]
    ), "The sums over the first variable must match the first margins."
    sum_over_var2 = (
        df_obs.groupby(["var1"])
        .agg({"raked_value": "sum"})
        .reset_index()
        .merge(df_margins_2, on="var1")
    )
    assert np.allclose(
        sum_over_var2["raked_value"], sum_over_var2["value_agg_over_var2"]
    ), "The sums over the second variable must match the second margins."


def test_run_raking_3D_weights(example_3D_draws):
    df_obs = example_3D_draws.df_obs
    df_obs = (
        df_obs.groupby(["var1", "var2", "var3"])
        .agg({"value": ["mean", "std"]})
        .reset_index()
    )
    df_obs.columns = [" ".join(col).strip() for col in df_obs.columns.values]
    df_obs.rename(
        columns={"value mean": "value", "value std": "weight"}, inplace=True
    )
    df_obs["lower"] = 1.8
    df_obs["upper"] = 3.2
    df_margins_1 = example_3D_draws.df_margins_1
    df_margins_1 = (
        df_margins_1.groupby(["var2", "var3"])
        .agg({"value_agg_over_var1": "mean"})
        .reset_index()
    )
    df_margins_2 = example_3D_draws.df_margins_2
    df_margins_2 = (
        df_margins_2.groupby(["var1", "var3"])
        .agg({"value_agg_over_var2": "mean"})
        .reset_index()
    )
    df_margins_3 = example_3D_draws.df_margins_3
    df_margins_3 = (
        df_margins_3.groupby(["var1", "var2"])
        .agg({"value_agg_over_var3": "mean"})
        .reset_index()
    )
    (df_obs, Dphi_y, Dphi_s, sigma) = run_raking(
        dim=3,
        df_obs=df_obs,
        df_margins=[
            df_margins_1,
            df_margins_2,
            df_margins_3,
        ],
        var_names=["var1", "var2", "var3"],
        cov_mat=False,
        method="logit",
        weights="weight",
        lower="lower",
        upper="upper",
    )
    sum_over_var1 = (
        df_obs.groupby(["var2", "var3"])
        .agg({"raked_value": "sum"})
        .reset_index()
        .merge(df_margins_1, on=["var2", "var3"])
    )
    assert np.allclose(
        sum_over_var1["raked_value"], sum_over_var1["value_agg_over_var1"]
    ), "The sums over the first variable must match the first margins."
    sum_over_var2 = (
        df_obs.groupby(["var1", "var3"])
        .agg({"raked_value": "sum"})
        .reset_index()
        .merge(df_margins_2, on=["var1", "var3"])
    )
    assert np.allclose(
        sum_over_var2["raked_value"], sum_over_var2["value_agg_over_var2"]
    ), "The sums over the second variable must match the second margins."
    sum_over_var3 = (
        df_obs.groupby(["var1", "var2"])
        .agg({"raked_value": "sum"})
        .reset_index()
        .merge(df_margins_3, on=["var1", "var2"])
    )
    assert np.allclose(
        sum_over_var3["raked_value"], sum_over_var3["value_agg_over_var3"]
    ), "The sums over the third variable must match the third margins."


def test_run_raking_USHD_weights(example_USHD_draws):
    df_obs = example_USHD_draws.df_obs
    df_obs = (
        df_obs.groupby(["cause", "race", "county", "upper"])
        .agg({"value": ["mean", "std"]})
        .reset_index()
    )
    df_obs.columns = [" ".join(col).strip() for col in df_obs.columns.values]
    df_obs.rename(
        columns={"value mean": "value", "value std": "weight"}, inplace=True
    )
    df_obs["lower"] = 0.0
    df_margins = example_USHD_draws.df_margins
    df_margins = (
        df_margins.groupby(["cause"])
        .agg({"value_agg_over_race_county": "mean"})
        .reset_index()
    )
    (df_obs, Dphi_y, Dphi_s, sigma) = run_raking(
        dim="USHD",
        df_obs=df_obs,
        df_margins=[df_margins],
        var_names=None,
        margin_names=["_all"],
        cov_mat=False,
        method="logit",
        weights="weight",
        lower="lower",
        upper="upper",
    )
    sum_over_cause = (
        df_obs.loc[df_obs.cause != "_all"]
        .groupby(["race", "county"])
        .agg({"raked_value": "sum"})
        .reset_index()
        .merge(df_obs.loc[df_obs.cause == "_all"], on=["race", "county"])
    )
    assert np.allclose(
        sum_over_cause["raked_value_x"],
        sum_over_cause["raked_value_y"],
        atol=1.0e-4,
    ), "The sums over the cause must match the all causes deaths."
    sum_over_race = (
        df_obs.loc[df_obs.race != 0]
        .groupby(["cause", "county"])
        .agg({"raked_value": "sum"})
        .reset_index()
        .merge(df_obs.loc[df_obs.race == 0], on=["cause", "county"])
    )
    assert np.allclose(
        sum_over_race["raked_value_x"],
        sum_over_race["raked_value_y"],
        atol=1.0e-4,
    ), "The sums over the race must match the all races deaths."
    sum_over_race_county = (
        df_obs.loc[df_obs.race != 0]
        .groupby(["cause"])
        .agg({"raked_value": "sum"})
        .reset_index()
        .merge(df_margins, on=["cause"])
    )
    assert np.allclose(
        sum_over_race_county["raked_value"],
        sum_over_race_county["value_agg_over_race_county"],
        atol=1.0e-5,
    ), "The sums over race and county must match the GBD values."
