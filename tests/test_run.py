import pytest
import numpy as np
import pandas as pd
from raking.run_raking import RakingData


def test_run_raking_1D(example_1D):
    data = RakingData(
        df_obs=example_1D.df_obs,
        df_margins=[example_1D.df_margin],
        var_names=["var1"],
    )
    data.rake()
    assert np.allclose(
        data.df_obs["raked_value"].sum(),
        example_1D.df_margin["value_agg_over_var1"].iloc[0],
    ), "The raked values do not sum to the margin."


def test_run_raking_2D(example_2D):
    data = RakingData(
        df_obs=example_2D.df_obs,
        df_margins=[example_2D.df_margins_1, example_2D.df_margins_2],
        var_names=["var1", "var2"],
    )
    data.rake()
    sum_over_var1 = (
        data.df_obs.groupby(["var2"])
        .agg({"raked_value": "sum"})
        .reset_index()
        .merge(example_2D.df_margins_1, on="var2")
    )
    assert np.allclose(
        sum_over_var1["raked_value"], sum_over_var1["value_agg_over_var1"]
    ), "The sums over the first variable must match the first margins."
    sum_over_var2 = (
        data.df_obs.groupby(["var1"])
        .agg({"raked_value": "sum"})
        .reset_index()
        .merge(example_2D.df_margins_2, on="var1")
    )
    assert np.allclose(
        sum_over_var2["raked_value"], sum_over_var2["value_agg_over_var2"]
    ), "The sums over the second variable must match the second margins."


def test_run_raking_3D(example_3D):
    data = RakingData(
        df_obs=example_3D.df_obs,
        df_margins=[
            example_3D.df_margins_1,
            example_3D.df_margins_2,
            example_3D.df_margins_3,
        ],
        var_names=["var1", "var2", "var3"],
    )
    data.rake()
    sum_over_var1 = (
        data.df_obs.groupby(["var2", "var3"])
        .agg({"raked_value": "sum"})
        .reset_index()
        .merge(example_3D.df_margins_1, on=["var2", "var3"])
    )
    assert np.allclose(
        sum_over_var1["raked_value"], sum_over_var1["value_agg_over_var1"]
    ), "The sums over the first variable must match the first margins."
    sum_over_var2 = (
        data.df_obs.groupby(["var1", "var3"])
        .agg({"raked_value": "sum"})
        .reset_index()
        .merge(example_3D.df_margins_2, on=["var1", "var3"])
    )
    assert np.allclose(
        sum_over_var2["raked_value"], sum_over_var2["value_agg_over_var2"]
    ), "The sums over the second variable must match the second margins."
    sum_over_var3 = (
        data.df_obs.groupby(["var1", "var2"])
        .agg({"raked_value": "sum"})
        .reset_index()
        .merge(example_3D.df_margins_3, on=["var1", "var2"])
    )
    assert np.allclose(
        sum_over_var3["raked_value"], sum_over_var3["value_agg_over_var3"]
    ), "The sums over the third variable must match the third margins."


def test_run_raking_USHD(example_USHD):
    data = RakingData(
        df_obs=example_USHD.df_obs,
        df_margins=[example_USHD.df_margins],
        use_case="USHD",
    )
    data.rake()
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
        .merge(example_USHD.df_margins, on=["cause"])
    )
    assert np.allclose(
        sum_over_race_county["raked_value"],
        sum_over_race_county["value_agg_over_race_county"],
        atol=1.0e-5,
    ), "The sums over race and county must match the GBD values."


def test_run_raking_1D_variance(example_1D_draws):
    df_obs = example_1D_draws.df_obs
    df_obs = (
        df_obs.groupby(["var1"]).agg({"value": ["mean", "var"]}).reset_index()
    )
    df_obs.columns = [" ".join(col).strip() for col in df_obs.columns.values]
    df_obs.rename(
        columns={"value mean": "value", "value var": "variance"}, inplace=True
    )
    df_margin = example_1D_draws.df_margin
    df_margin = pd.DataFrame(
        data={
            "value_agg_over_var1": [np.mean(df_margin.value_agg_over_var1)],
            "variance": [np.var(df_margin.value_agg_over_var1)],
        }
    )
    data = RakingData(
        df_obs=df_obs,
        df_margins=[df_margin],
        var_names=["var1"],
    )
    data.rake()
    data.compute_variance("variance")
    assert np.allclose(
        data.df_obs["raked_value"].sum(),
        df_margin["value_agg_over_var1"].iloc[0],
    ), "The raked values do not sum to the margin."


def test_run_raking_2D_variance(example_2D_draws):
    df_obs = example_2D_draws.df_obs
    df_obs = (
        df_obs.groupby(["var1", "var2"])
        .agg({"value": ["mean", "var"]})
        .reset_index()
    )
    df_obs.columns = [" ".join(col).strip() for col in df_obs.columns.values]
    df_obs.rename(
        columns={"value mean": "value", "value var": "variance"}, inplace=True
    )
    df_margins_1 = example_2D_draws.df_margins_1
    df_margins_2 = example_2D_draws.df_margins_2
    df_margins_1 = (
        df_margins_1.groupby(["var2"])
        .agg({"value_agg_over_var1": ["mean", "var"]})
        .reset_index()
    )
    df_margins_1.columns = [
        " ".join(col).strip() for col in df_margins_1.columns.values
    ]
    df_margins_1.rename(
        columns={
            "value_agg_over_var1 mean": "value_agg_over_var1",
            "value_agg_over_var1 var": "variance",
        },
        inplace=True,
    )
    df_margins_2 = (
        df_margins_2.groupby(["var1"])
        .agg({"value_agg_over_var2": ["mean", "var"]})
        .reset_index()
    )
    df_margins_2.columns = [
        " ".join(col).strip() for col in df_margins_2.columns.values
    ]
    df_margins_2.rename(
        columns={
            "value_agg_over_var2 mean": "value_agg_over_var2",
            "value_agg_over_var2 var": "variance",
        },
        inplace=True,
    )
    data = RakingData(
        df_obs=df_obs,
        df_margins=[df_margins_1, df_margins_2],
        var_names=["var1", "var2"],
    )
    data.rake()
    data.compute_variance("variance")
    sum_over_var1 = (
        data.df_obs.groupby(["var2"])
        .agg({"raked_value": "sum"})
        .reset_index()
        .merge(df_margins_1, on="var2")
    )
    assert np.allclose(
        sum_over_var1["raked_value"], sum_over_var1["value_agg_over_var1"]
    ), "The sums over the first variable must match the first margins."
    sum_over_var2 = (
        data.df_obs.groupby(["var1"])
        .agg({"raked_value": "sum"})
        .reset_index()
        .merge(df_margins_2, on="var1")
    )
    assert np.allclose(
        sum_over_var2["raked_value"], sum_over_var2["value_agg_over_var2"]
    ), "The sums over the second variable must match the second margins."


def test_run_raking_3D_variance(example_3D_draws):
    df_obs = example_3D_draws.df_obs
    df_obs = (
        df_obs.groupby(["var1", "var2", "var3"])
        .agg({"value": ["mean", "var"]})
        .reset_index()
    )
    df_obs.columns = [" ".join(col).strip() for col in df_obs.columns.values]
    df_obs.rename(
        columns={"value mean": "value", "value var": "variance"}, inplace=True
    )
    df_margins_1 = example_3D_draws.df_margins_1
    df_margins_2 = example_3D_draws.df_margins_2
    df_margins_3 = example_3D_draws.df_margins_3
    df_margins_1 = (
        df_margins_1.groupby(["var2", "var3"])
        .agg({"value_agg_over_var1": ["mean", "var"]})
        .reset_index()
    )
    df_margins_1.columns = [
        " ".join(col).strip() for col in df_margins_1.columns.values
    ]
    df_margins_1.rename(
        columns={
            "value_agg_over_var1 mean": "value_agg_over_var1",
            "value_agg_over_var1 var": "variance",
        },
        inplace=True,
    )
    df_margins_2 = (
        df_margins_2.groupby(["var1", "var3"])
        .agg({"value_agg_over_var2": ["mean", "var"]})
        .reset_index()
    )
    df_margins_2.columns = [
        " ".join(col).strip() for col in df_margins_2.columns.values
    ]
    df_margins_2.rename(
        columns={
            "value_agg_over_var2 mean": "value_agg_over_var2",
            "value_agg_over_var2 var": "variance",
        },
        inplace=True,
    )
    df_margins_3 = (
        df_margins_3.groupby(["var1", "var2"])
        .agg({"value_agg_over_var3": ["mean", "var"]})
        .reset_index()
    )
    df_margins_3.columns = [
        " ".join(col).strip() for col in df_margins_3.columns.values
    ]
    df_margins_3.rename(
        columns={
            "value_agg_over_var3 mean": "value_agg_over_var3",
            "value_agg_over_var3 var": "variance",
        },
        inplace=True,
    )
    data = RakingData(
        df_obs=df_obs,
        df_margins=[df_margins_1, df_margins_2, df_margins_3],
        var_names=["var1", "var2", "var3"],
    )
    data.rake()
    data.compute_variance("variance")
    sum_over_var1 = (
        data.df_obs.groupby(["var2", "var3"])
        .agg({"raked_value": "sum"})
        .reset_index()
        .merge(df_margins_1, on=["var2", "var3"])
    )
    assert np.allclose(
        sum_over_var1["raked_value"], sum_over_var1["value_agg_over_var1"]
    ), "The sums over the first variable must match the first margins."
    sum_over_var2 = (
        data.df_obs.groupby(["var1", "var3"])
        .agg({"raked_value": "sum"})
        .reset_index()
        .merge(df_margins_2, on=["var1", "var3"])
    )
    assert np.allclose(
        sum_over_var2["raked_value"], sum_over_var2["value_agg_over_var2"]
    ), "The sums over the second variable must match the second margins."
    sum_over_var3 = (
        data.df_obs.groupby(["var1", "var2"])
        .agg({"raked_value": "sum"})
        .reset_index()
        .merge(df_margins_3, on=["var1", "var2"])
    )
    assert np.allclose(
        sum_over_var3["raked_value"], sum_over_var3["value_agg_over_var3"]
    ), "The sums over the third variable must match the third margins."


def test_run_raking_USHD_variance(example_USHD_draws):
    df_obs = example_USHD_draws.df_obs
    df_obs = (
        df_obs.groupby(["cause", "race", "county"])
        .agg({"value": ["mean", "var"]})
        .reset_index()
    )
    df_obs.columns = [" ".join(col).strip() for col in df_obs.columns.values]
    df_obs.rename(
        columns={"value mean": "value", "value var": "variance"}, inplace=True
    )
    df_margins = example_USHD_draws.df_margins
    df_margins = (
        df_margins.groupby(["cause"])
        .agg({"value_agg_over_race_county": ["mean", "var"]})
        .reset_index()
    )
    df_margins.columns = [
        " ".join(col).strip() for col in df_margins.columns.values
    ]
    df_margins.rename(
        columns={
            "value_agg_over_race_county mean": "value_agg_over_race_county",
            "value_agg_over_race_county var": "variance",
        },
        inplace=True,
    )
    data = RakingData(
        df_obs=df_obs,
        df_margins=[df_margins],
        use_case="USHD",
    )
    data.rake()
    data.compute_variance("variance")
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
        .merge(df_margins, on=["cause"])
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
    data = RakingData(
        df_obs=df_obs,
        df_margins=[df_margin],
        var_names=["var1"],
        weights="weight",
        lower="lower",
        upper="upper",
    )
    data.rake(method="logit")
    assert np.allclose(
        data.df_obs["raked_value"].sum(),
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
    data = RakingData(
        df_obs=df_obs,
        df_margins=[
            df_margins_1,
            df_margins_2,
        ],
        var_names=["var1", "var2"],
        weights="weight",
        lower="lower",
        upper="upper",
    )
    data.rake(method="logit")
    sum_over_var1 = (
        data.df_obs.groupby(["var2"])
        .agg({"raked_value": "sum"})
        .reset_index()
        .merge(df_margins_1, on="var2")
    )
    assert np.allclose(
        sum_over_var1["raked_value"], sum_over_var1["value_agg_over_var1"]
    ), "The sums over the first variable must match the first margins."
    sum_over_var2 = (
        data.df_obs.groupby(["var1"])
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
    data = RakingData(
        df_obs=df_obs,
        df_margins=[
            df_margins_1,
            df_margins_2,
            df_margins_3,
        ],
        var_names=["var1", "var2", "var3"],
        weights="weight",
        lower="lower",
        upper="upper",
    )
    data.rake(method="logit")
    sum_over_var1 = (
        data.df_obs.groupby(["var2", "var3"])
        .agg({"raked_value": "sum"})
        .reset_index()
        .merge(df_margins_1, on=["var2", "var3"])
    )
    assert np.allclose(
        sum_over_var1["raked_value"], sum_over_var1["value_agg_over_var1"]
    ), "The sums over the first variable must match the first margins."
    sum_over_var2 = (
        data.df_obs.groupby(["var1", "var3"])
        .agg({"raked_value": "sum"})
        .reset_index()
        .merge(df_margins_2, on=["var1", "var3"])
    )
    assert np.allclose(
        sum_over_var2["raked_value"], sum_over_var2["value_agg_over_var2"]
    ), "The sums over the second variable must match the second margins."
    sum_over_var3 = (
        data.df_obs.groupby(["var1", "var2"])
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
    data = RakingData(
        df_obs=df_obs,
        df_margins=[df_margins],
        use_case="USHD",
        weights="weight",
        lower="lower",
        upper="upper",
    )
    data.rake(method="logit")
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
        .merge(df_margins, on=["cause"])
    )
    assert np.allclose(
        sum_over_race_county["raked_value"],
        sum_over_race_county["value_agg_over_race_county"],
        atol=1.0e-5,
    ), "The sums over race and county must match the GBD values."
