import pytest
import numpy as np
import pandas as pd
from raking.experimental import DataBuilder
from raking.experimental import DataBuilderParallel
from raking.experimental import DualSolver
from raking.experimental import DualSolverParallel


def test_parallel(example_1D_draws):
    df_obs = example_1D_draws.df_obs
    df_margin = example_1D_draws.df_margin
    df_obs["weights"] = 1.0
    df_margin["var1"] = -1
    df_margin["weights"] = np.inf
    df_margin.rename(columns={"value_agg_over_var1": "value"}, inplace=True)
    df = pd.concat([df_obs, df_margin])

    df = df.loc[df.draws < 10.5]

    # Loop on the draws
    draws = df.draws.unique().tolist()
    df_raked = []
    for draw in draws:
        df_loc = df.loc[df.draws == draw]
        data_builder = DataBuilder(
            dim_specs={"var1": -1}, value="value", weights="weights"
        )
        data = data_builder.build(df_loc)
        solver = DualSolver(distance="entropic", data=data)
        soln = solver.solve()
        soln["draws"] = draw
        df_raked.append(soln)
    df_raked = pd.concat(df_raked)

    # Check the results for the loop on the draws
    sum_over_var1 = (
        df_raked.groupby(["draws"])
        .agg({"soln": "sum"})
        .merge(df.loc[df["var1"] == -1], on=["draws"], how="inner")
    )
    assert np.allclose(
        sum_over_var1["soln"].to_numpy(), sum_over_var1["value"].to_numpy()
    ), "For the loop, the raked values do not sum to the margin."

    # Use parallelization
    # The matrix in the objective of the dual has shape (3000, 1000) and full rank.
    # The solver converges correctly.
    data_builder = DataBuilderParallel(
        dim_specs={"var1": -1},
        dim_parallel=["draws"],
        value="value",
        weights="weights",
    )
    data = data_builder.build(df)
    solver = DualSolverParallel(distance="entropic", data=data)
    df_raked_parallel = solver.solve()

    # Check the results for the parallelization
    sum_over_var1 = (
        df_raked_parallel.groupby(["draws"])
        .agg({"soln": "sum"})
        .merge(df.loc[df["var1"] == -1], on=["draws"], how="inner")
    )
    assert np.allclose(
        sum_over_var1["soln"].to_numpy(), sum_over_var1["value"].to_numpy()
    ), "For the parallelization, the raked values do not sum to the margin."

    # Compare results
    df_both = pd.merge(
        df_raked,
        df_raked_parallel,
        on=["var1", "draws"],
        how="inner",
    )
    assert np.allclose(df_both["soln_x"], df_both["soln_y"]), (
        "The two rakings must give the same results."
    )
