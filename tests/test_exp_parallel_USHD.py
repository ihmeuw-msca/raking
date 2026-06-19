import pytest
import numpy as np
import pandas as pd
from raking.experimental import DataBuilder
from raking.experimental import DataBuilderParallel
from raking.experimental import DualSolver
from raking.experimental import DualSolverParallel


def test_parallel(example_USHD_draws):
    df_obs = example_USHD_draws.df_obs
    df_margin = example_USHD_draws.df_margins
    df_obs["weights"] = 1.0
    df_obs.drop(columns=["upper"], inplace=True)
    df_obs["cause"] = (
        df_obs["cause"]
        .map({"_all": -1, "_comm": 1, "_inj": 2, "_ncd": 3})
        .astype("int64")
    )
    df_obs["race"] = df_obs["race"].replace(1, -1)
    df_margin["race"] = -1
    df_margin["county"] = -1
    df_margin["weights"] = np.inf
    df_margin.rename(
        columns={"value_agg_over_race_county": "value"}, inplace=True
    )
    df_margin["cause"] = (
        df_margin["cause"]
        .map({"_all": -1, "_comm": 1, "_inj": 2, "_ncd": 3})
        .astype("int64")
    )
    df = pd.concat([df_obs, df_margin])

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
    sum_over_race_county = (
        df_raked.groupby(["cause", "draws"], observed=True)
        .agg({"soln": "sum"})
        .reset_index()
        .merge(df_margin, on=["cause", "draws"])
    )
    assert np.allclose(
        sum_over_race_county["soln"],
        sum_over_race_county["value"],
        atol=1.0e-3,
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
    sum_over_race_county = (
        df_raked_parallel.groupby(["cause", "draws"], observed=True)
        .agg({"soln": "sum"})
        .reset_index()
        .merge(df_margin, on=["cause", "draws"])
    )
    assert np.allclose(
        sum_over_race_county["soln"], sum_over_race_county["value"]
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
