import pytest
import numpy as np
import pandas as pd
from raking.experimental import DataBuilder
from raking.experimental import DataBuilderParallel
from raking.experimental import DualSolver
from raking.experimental import DualSolverParallel


def test_parallel(example_USHD_lower_draws):
    df_obs = example_USHD_lower_draws.df_obs
    df_margin_cause = example_USHD_lower_draws.df_margins_cause
    df_margin_county = example_USHD_lower_draws.df_margins_county
    df_margin_all_causes = example_USHD_lower_draws.df_margins_all_causes
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

    df = df.loc[(df["cause"] != -1) | (df["race"] != 2) | (df["county"] != 301)]

    # Loop on the draws
    draws = df.draws.unique().tolist()
    df_raked = []
    for draw in draws:
        df_loc = df.loc[df.draws == draw]
        data_builder = DataBuilder(
            dim_specs={"cause": -1, "race": -1, "county": -1},
            value="value",
            weights="weights",
        )
        data = data_builder.build(df_loc)
        solver = DualSolver(distance="entropic", data=data)
        soln = solver.solve()
        soln["draws"] = draw
        df_raked.append(soln)
    df_raked = pd.concat(df_raked)

    # Check the results for the loop on the draws
    sum_over_cause = (
        df_raked.groupby(["race", "county", "draws"], observed=True)
        .agg({"soln": "sum"})
        .reset_index()
        .merge(df_margin_all_causes, on=["race", "county", "draws"])
    )
    sum_over_cause_race = (
        df_raked.groupby(["county", "draws"], observed=True)
        .agg({"soln": "sum"})
        .reset_index()
        .merge(df_margin_county, on=["county", "draws"])
    )
    sum_over_race_county = (
        df_raked.groupby(["cause", "draws"], observed=True)
        .agg({"soln": "sum"})
        .reset_index()
        .merge(df_margin_cause, on=["cause", "draws"])
    )
    assert np.allclose(
        sum_over_cause["soln"],
        sum_over_cause["value"],
        atol=1.0e-4,
    ), "For the loop, the sums over cause must match the margins."
    assert np.allclose(
        sum_over_cause_race["soln"],
        sum_over_cause_race["value"],
        atol=1.0e-4,
    ), "For the loop, the sums over cause and race must match the margins."
    assert np.allclose(
        sum_over_race_county["soln"],
        sum_over_race_county["value"],
        atol=1.0e-5,
    ), "For the loop, the sums over race and county must match the GBD values."

    # Use parallelization
    data_builder = DataBuilderParallel(
        dim_specs={"cause": -1, "race": -1, "county": -1},
        dim_parallel=["draws"],
        value="value",
        weights="weights",
    )
    data = data_builder.build(df)
    solver = DualSolverParallel(distance="entropic", data=data)
    df_raked_parallel = solver.solve()

    # Check the results for the parallelization
    sum_over_cause = (
        df_raked_parallel.groupby(["race", "county", "draws"], observed=True)
        .agg({"soln": "sum"})
        .reset_index()
        .merge(df_margin_all_causes, on=["race", "county", "draws"])
    )
    sum_over_cause_race = (
        df_raked_parallel.groupby(["county", "draws"], observed=True)
        .agg({"soln": "sum"})
        .reset_index()
        .merge(df_margin_county, on=["county", "draws"])
    )
    sum_over_race_county = (
        df_raked_parallel.groupby(["cause", "draws"], observed=True)
        .agg({"soln": "sum"})
        .reset_index()
        .merge(df_margin_cause, on=["cause", "draws"])
    )
    assert np.allclose(
        sum_over_cause["soln"],
        sum_over_cause["value"],
        atol=1.0e-5,
    ), "For the parallelization, the sums over cause must match the margins."
    assert np.allclose(
        sum_over_cause_race["soln"], sum_over_cause_race["value"]
    ), (
        "For the parallelization, the sums over cause and race must match the margins."
    )
    assert np.allclose(
        sum_over_race_county["soln"],
        sum_over_race_county["value"],
    ), (
        "For the parallelization, the sums over race and county must match the GBD values."
    )

    # Compare results
    df_both = pd.merge(
        df_raked,
        df_raked_parallel,
        on=["cause", "race", "county", "draws"],
        how="inner",
    )
    assert np.allclose(df_both["soln_x"], df_both["soln_y"], 1.0e-2), (
        "The two rakings must give the same results."
    )
