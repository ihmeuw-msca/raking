import pytest
import numpy as np
import pandas as pd
from raking.experimental import DataBuilder
from raking.experimental import DualSolver


def test_exp_raking_1D(example_1D):
    df_obs = example_1D.df_obs
    df_margin = example_1D.df_margin
    df_obs["weights"] = 1.0
    df_margin["var1"] = -1
    df_margin["weights"] = np.inf
    df_margin.rename(columns={"value_agg_over_var1": "value"}, inplace=True)
    df = pd.concat([df_obs, df_margin])
    data_builder = DataBuilder(
        dim_specs={"var1": -1}, value="value", weights="weights"
    )
    data = data_builder.build(df)
    solver = DualSolver(distance="entropic", data=data)
    df_raked = solver.solve()
    assert np.allclose(df_raked["soln"].sum(), df_margin["value"].iloc[0]), (
        "The raked values do not sum to the margin."
    )


def test_exp_raking_2D(example_2D):
    df_obs = example_2D.df_obs
    df_margins_1 = example_2D.df_margins_1
    df_margins_2 = example_2D.df_margins_2
    df_obs["weights"] = 1.0
    df_margins_1["var1"] = -1
    df_margins_1["weights"] = np.inf
    df_margins_1.rename(columns={"value_agg_over_var1": "value"}, inplace=True)
    df_margins_2["var2"] = -1
    df_margins_2["weights"] = np.inf
    df_margins_2.rename(columns={"value_agg_over_var2": "value"}, inplace=True)
    df = pd.concat([df_obs, df_margins_1, df_margins_2])
    data_builder = DataBuilder(
        dim_specs={"var1": -1, "var2": -1}, value="value", weights="weights"
    )
    data = data_builder.build(df)
    solver = DualSolver(distance="entropic", data=data)
    df_raked = solver.solve()
    sum_over_var1 = (
        df_raked.groupby(["var2"])
        .agg({"soln": "sum"})
        .reset_index()
        .merge(df_margins_1, on="var2")
    )
    assert np.allclose(sum_over_var1["soln"], sum_over_var1["value"]), (
        "The sums over the first variable must match the first margins."
    )
    sum_over_var2 = (
        df_raked.groupby(["var1"])
        .agg({"soln": "sum"})
        .reset_index()
        .merge(df_margins_2, on="var1")
    )
    assert np.allclose(sum_over_var2["soln"], sum_over_var2["value"]), (
        "The sums over the second variable must match the second margins."
    )


def test_exp_raking_3D(example_3D):
    df_obs = example_3D.df_obs
    df_margins_1 = example_3D.df_margins_1
    df_margins_2 = example_3D.df_margins_2
    df_margins_3 = example_3D.df_margins_3
    df_obs["weights"] = 1.0
    df_margins_1["var1"] = -1
    df_margins_1["weights"] = np.inf
    df_margins_1.rename(columns={"value_agg_over_var1": "value"}, inplace=True)
    df_margins_2["var2"] = -1
    df_margins_2["weights"] = np.inf
    df_margins_2.rename(columns={"value_agg_over_var2": "value"}, inplace=True)
    df_margins_3["var3"] = -1
    df_margins_3["weights"] = np.inf
    df_margins_3.rename(columns={"value_agg_over_var3": "value"}, inplace=True)
    df = pd.concat([df_obs, df_margins_1, df_margins_2, df_margins_3])
    data_builder = DataBuilder(
        dim_specs={"var1": -1, "var2": -1, "var3": -1},
        value="value",
        weights="weights",
    )
    data = data_builder.build(df)
    solver = DualSolver(distance="entropic", data=data)
    df_raked = solver.solve()
    sum_over_var1 = (
        df_raked.groupby(["var2", "var3"])
        .agg({"soln": "sum"})
        .reset_index()
        .merge(df_margins_1, on=["var2", "var3"])
    )
    assert np.allclose(sum_over_var1["soln"], sum_over_var1["value"]), (
        "The sums over the first variable must match the first margins."
    )
    sum_over_var2 = (
        df_raked.groupby(["var1", "var3"])
        .agg({"soln": "sum"})
        .reset_index()
        .merge(df_margins_2, on=["var1", "var3"])
    )
    assert np.allclose(sum_over_var2["soln"], sum_over_var2["value"]), (
        "The sums over the second variable must match the second margins."
    )
    sum_over_var3 = (
        df_raked.groupby(["var1", "var2"])
        .agg({"soln": "sum"})
        .reset_index()
        .merge(df_margins_3, on=["var1", "var2"])
    )
    assert np.allclose(sum_over_var3["soln"], sum_over_var3["value"]), (
        "The sums over the third variable must match the third margins."
    )


def test_exp_raking_USHD(example_USHD):
    df_obs = example_USHD.df_obs
    df_margin = example_USHD.df_margins
    df_obs["weights"] = 1.0
    df_obs.replace({"cause": "_all", "race": 1}, -1, inplace=True)
    df_obs.drop(columns=["upper"], inplace=True)
    df_obs.replace({"cause": {"_comm": 1, "_inj": 2, "_ncd": 3}}, inplace=True)
    df_margin["race"] = -1
    df_margin["county"] = -1
    df_margin["weights"] = np.inf
    df_margin.rename(
        columns={"value_agg_over_race_county": "value"}, inplace=True
    )
    df_margin.replace(
        {"cause": {"_all": -1, "_comm": 1, "_inj": 2, "_ncd": 3}}, inplace=True
    )
    df = pd.concat([df_obs, df_margin])
    df = df.astype({"cause": "int64"})
    data_builder = DataBuilder(
        dim_specs={"cause": -1, "race": -1, "county": -1},
        value="value",
        weights="weights",
    )
    data = data_builder.build(df)
    solver = DualSolver(distance="entropic", data=data)
    df_raked = solver.solve()
    sum_over_race_county = (
        df_raked.groupby(["cause"], observed=True)
        .agg({"soln": "sum"})
        .reset_index()
        .merge(df_margin, on=["cause"])
    )
    assert np.allclose(
        sum_over_race_county["soln"],
        sum_over_race_county["value"],
        atol=1.0e-5,
    ), "The sums over race and county must match the GBD values."


def test_exp_raking_USHD_lower(example_USHD_lower):
    df_obs = example_USHD_lower.df_obs
    df_margin_cause = example_USHD_lower.df_margins_cause
    df_margin_county = example_USHD_lower.df_margins_county
    df_margin_all_causes = example_USHD_lower.df_margins_all_causes
    df_obs["weights"] = 1.0
    df_obs.replace({"race": 1}, -1, inplace=True)
    df_obs.drop(columns=["upper"], inplace=True)
    df_obs.replace(
        {"cause": {"_intent": 1, "_unintent": 2, "inj_trans": 3}}, inplace=True
    )
    df_margin_cause["race"] = -1
    df_margin_cause["county"] = -1
    df_margin_cause["weights"] = np.inf
    df_margin_cause.rename(
        columns={"value_agg_over_race_county": "value"}, inplace=True
    )
    df_margin_cause.replace(
        {"cause": {"_intent": 1, "_unintent": 2, "inj_trans": 3}}, inplace=True
    )
    df_margin_county["cause"] = -1
    df_margin_county["race"] = -1
    df_margin_county["weights"] = np.inf
    df_margin_county.rename(
        columns={"value_agg_over_cause_race": "value"}, inplace=True
    )
    df_margin_all_causes["cause"] = -1
    df_margin_all_causes["weights"] = np.inf
    df_margin_all_causes.rename(
        columns={"value_agg_over_cause": "value"}, inplace=True
    )
    df = pd.concat(
        [df_obs, df_margin_cause, df_margin_county, df_margin_all_causes]
    )
    df = df.astype({"cause": "int64"})
    data_builder = DataBuilder(
        dim_specs={"cause": -1, "race": -1, "county": -1},
        value="value",
        weights="weights",
    )
    data = data_builder.build(df)
    solver = DualSolver(distance="entropic", data=data)
    df_raked = solver.solve()
    sum_over_cause = (
        df_raked.groupby(["race", "county"], observed=True)
        .agg({"soln": "sum"})
        .reset_index()
        .merge(df_margin_all_causes, on=["race", "county"])
    )
    assert np.allclose(
        sum_over_cause["soln"],
        sum_over_cause["value"],
        atol=1.0e-4,
    ), "The sums over the cause must match the all causes deaths."
    sum_over_cause_race = (
        df_raked.groupby(["county"], observed=True)
        .agg({"soln": "sum"})
        .reset_index()
        .merge(df_margin_county, on=["county"])
    )
    assert np.allclose(
        sum_over_cause_race["soln"],
        sum_over_cause_race["value"],
        atol=1.0e-4,
    ), (
        "The sums over the cause and race must match the all causes all races deaths."
    )
    sum_over_race_county = (
        df_raked.groupby(["cause"], observed=True)
        .agg({"soln": "sum"})
        .reset_index()
        .merge(df_margin_cause, on=["cause"])
    )
    assert np.allclose(
        sum_over_race_county["soln"],
        sum_over_race_county["value"],
        atol=1.0e-5,
    ), "The sums over race and county must match the GBD values."


def test_exp_raking_1D_weights(example_1D_draws):
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
    df_margin["var1"] = -1
    df_margin["weight"] = np.inf
    df_margin.rename(columns={"value_agg_over_var1": "value"}, inplace=True)
    df = pd.concat([df_obs, df_margin])
    data_builder = DataBuilder(
        dim_specs={"var1": -1},
        value="value",
        weights="weight",
        bounds=("lower", "upper"),
    )
    data = data_builder.build(df)
    solver = DualSolver(distance="logistic", data=data)
    df_raked = solver.solve()
    assert np.allclose(df_raked["soln"].sum(), df_margin["value"].iloc[0]), (
        "The raked values do not sum to the margin."
    )


def test_exp_raking_2D_weights(example_2D_draws):
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
    df_margins_1["var1"] = -1
    df_margins_1["weight"] = np.inf
    df_margins_1.rename(columns={"value_agg_over_var1": "value"}, inplace=True)
    df_margins_2["var2"] = -1
    df_margins_2["weight"] = np.inf
    df_margins_2.rename(columns={"value_agg_over_var2": "value"}, inplace=True)
    df = pd.concat([df_obs, df_margins_1, df_margins_2])
    data_builder = DataBuilder(
        dim_specs={"var1": -1, "var2": -1},
        value="value",
        weights="weight",
        bounds=("lower", "upper"),
    )
    data = data_builder.build(df)
    solver = DualSolver(distance="logistic", data=data)
    df_raked = solver.solve()
    sum_over_var1 = (
        df_raked.groupby(["var2"])
        .agg({"soln": "sum"})
        .reset_index()
        .merge(df_margins_1, on="var2")
    )
    assert np.allclose(sum_over_var1["soln"], sum_over_var1["value"]), (
        "The sums over the first variable must match the first margins."
    )
    sum_over_var2 = (
        df_raked.groupby(["var1"])
        .agg({"soln": "sum"})
        .reset_index()
        .merge(df_margins_2, on="var1")
    )
    assert np.allclose(sum_over_var2["soln"], sum_over_var2["value"]), (
        "The sums over the second variable must match the second margins."
    )


def test_exp_raking_3D_weights(example_3D_draws):
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
    df_margins_1["var1"] = -1
    df_margins_1["weight"] = np.inf
    df_margins_1.rename(columns={"value_agg_over_var1": "value"}, inplace=True)
    df_margins_2["var2"] = -1
    df_margins_2["weight"] = np.inf
    df_margins_2.rename(columns={"value_agg_over_var2": "value"}, inplace=True)
    df_margins_3["var3"] = -1
    df_margins_3["weight"] = np.inf
    df_margins_3.rename(columns={"value_agg_over_var3": "value"}, inplace=True)
    df = pd.concat([df_obs, df_margins_1, df_margins_2, df_margins_3])
    data_builder = DataBuilder(
        dim_specs={"var1": -1, "var2": -1, "var3": -1},
        value="value",
        weights="weight",
        bounds=("lower", "upper"),
    )
    data = data_builder.build(df)
    solver = DualSolver(distance="logistic", data=data)
    df_raked = solver.solve()
    sum_over_var1 = (
        df_raked.groupby(["var2", "var3"])
        .agg({"soln": "sum"})
        .reset_index()
        .merge(df_margins_1, on=["var2", "var3"])
    )
    assert np.allclose(sum_over_var1["soln"], sum_over_var1["value"]), (
        "The sums over the first variable must match the first margins."
    )
    sum_over_var2 = (
        df_raked.groupby(["var1", "var3"])
        .agg({"soln": "sum"})
        .reset_index()
        .merge(df_margins_2, on=["var1", "var3"])
    )
    assert np.allclose(sum_over_var2["soln"], sum_over_var2["value"]), (
        "The sums over the second variable must match the second margins."
    )
    sum_over_var3 = (
        df_raked.groupby(["var1", "var2"])
        .agg({"soln": "sum"})
        .reset_index()
        .merge(df_margins_3, on=["var1", "var2"])
    )
    assert np.allclose(sum_over_var3["soln"], sum_over_var3["value"]), (
        "The sums over the third variable must match the third margins."
    )


def test_exp_raking_USHD_weights(example_USHD_draws):
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
    df_obs.replace({"cause": "_all", "race": 1}, -1, inplace=True)
    df_obs.replace({"cause": {"_comm": 1, "_inj": 2, "_ncd": 3}}, inplace=True)
    df_margins["race"] = -1
    df_margins["county"] = -1
    df_margins["weight"] = np.inf
    df_margins.rename(
        columns={"value_agg_over_race_county": "value"}, inplace=True
    )
    df_margins.replace(
        {"cause": {"_all": -1, "_comm": 1, "_inj": 2, "_ncd": 3}}, inplace=True
    )
    df = pd.concat([df_obs, df_margins])
    df = df.astype({"cause": "int64"})
    data_builder = DataBuilder(
        dim_specs={"cause": -1, "race": -1, "county": -1},
        value="value",
        weights="weight",
        bounds=("lower", "upper"),
    )
    data = data_builder.build(df)
    solver = DualSolver(distance="logistic", data=data)
    df_raked = solver.solve()
    sum_over_race_county = (
        df_raked.groupby(["cause"], observed=True)
        .agg({"soln": "sum"})
        .reset_index()
        .merge(df_margins, on=["cause"])
    )
    assert np.allclose(
        sum_over_race_county["soln"],
        sum_over_race_county["value"],
        atol=1.0e-5,
    ), "The sums over race and county must match the GBD values."


def test_exp_raking_USHD_lower_weights(example_USHD_lower_draws):
    df_obs = example_USHD_lower_draws.df_obs
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
    df_margins_cause = example_USHD_lower_draws.df_margins_cause
    df_margins_cause = (
        df_margins_cause.groupby(["cause"])
        .agg({"value_agg_over_race_county": "mean"})
        .reset_index()
    )
    df_margins_county = example_USHD_lower_draws.df_margins_county
    df_margins_county = (
        df_margins_county.groupby(["county"])
        .agg({"value_agg_over_cause_race": "mean"})
        .reset_index()
    )
    df_margins_all_causes = example_USHD_lower_draws.df_margins_all_causes
    df_margins_all_causes = (
        df_margins_all_causes.groupby(["race", "county"])
        .agg({"value_agg_over_cause": "mean"})
        .reset_index()
    )
    df_obs.replace({"race": 1}, -1, inplace=True)
    df_obs.replace(
        {"cause": {"_intent": 1, "_unintent": 2, "inj_trans": 3}}, inplace=True
    )
    df_margins_cause["race"] = -1
    df_margins_cause["county"] = -1
    df_margins_cause["weight"] = np.inf
    df_margins_cause.rename(
        columns={"value_agg_over_race_county": "value"}, inplace=True
    )
    df_margins_cause.replace(
        {"cause": {"_intent": 1, "_unintent": 2, "inj_trans": 3}}, inplace=True
    )
    df_margins_county["cause"] = -1
    df_margins_county["race"] = -1
    df_margins_county["weight"] = np.inf
    df_margins_county.rename(
        columns={"value_agg_over_cause_race": "value"}, inplace=True
    )
    df_margins_all_causes["cause"] = -1
    df_margins_all_causes["weight"] = np.inf
    df_margins_all_causes.rename(
        columns={"value_agg_over_cause": "value"}, inplace=True
    )
    df = pd.concat(
        [df_obs, df_margins_cause, df_margins_county, df_margins_all_causes]
    )
    df = df.astype({"cause": "int64"})
    data_builder = DataBuilder(
        dim_specs={"cause": -1, "race": -1, "county": -1},
        value="value",
        weights="weight",
        bounds=("lower", "upper"),
    )
    data = data_builder.build(df)
    solver = DualSolver(distance="logistic", data=data)
    df_raked = solver.solve()
    sum_over_cause = (
        df_raked.groupby(["race", "county"], observed=True)
        .agg({"soln": "sum"})
        .reset_index()
        .merge(df_margins_all_causes, on=["race", "county"])
    )
    assert np.allclose(
        sum_over_cause["soln"],
        sum_over_cause["value"],
        atol=1.0e-4,
    ), "The sums over the cause must match the all causes deaths."
    sum_over_cause_race = (
        df_raked.groupby(["county"], observed=True)
        .agg({"soln": "sum"})
        .reset_index()
        .merge(df_margins_county, on=["county"])
    )
    assert np.allclose(
        sum_over_cause_race["soln"],
        sum_over_cause_race["value"],
        atol=1.0e-4,
    ), (
        "The sums over the cause and race must match the all causes all races deaths."
    )
    sum_over_race_county = (
        df_raked.groupby(["cause"], observed=True)
        .agg({"soln": "sum"})
        .reset_index()
        .merge(df_margins_cause, on=["cause"])
    )
    assert np.allclose(
        sum_over_race_county["soln"],
        sum_over_race_county["value"],
        atol=1.0e-5,
    ), "The sums over race and county must match the GBD values."
