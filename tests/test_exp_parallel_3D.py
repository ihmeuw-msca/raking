import pytest
import numpy as np
import pandas as pd
from raking.experimental import DataBuilder
from raking.experimental import DataBuilderParallel
from raking.experimental import DualSolver
from raking.experimental import DualSolverParallel


def test_parallel(example_3D_draws):
    df_obs = example_3D_draws.df_obs
    df_margins_1 = example_3D_draws.df_margins_1
    df_margins_2 = example_3D_draws.df_margins_2
    df_margins_3 = example_3D_draws.df_margins_3
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

    # Loop on the draws
    draws = df.draws.unique().tolist()
    df_raked = []
    for draw in draws:
        df_loc = df.loc[df.draws == draw]
        data_builder = DataBuilder(
            dim_specs={"var1": -1, "var2": -1, "var3": -1},
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
    sum_over_var1 = (
        df_raked.groupby(["var2", "var3", "draws"])
        .agg({"soln": "sum"})
        .reset_index()
        .merge(df_margins_1, on=["var2", "var3", "draws"])
    )
    assert np.allclose(sum_over_var1["soln"], sum_over_var1["value"], 1.0e-4), (
        "For the loop, the sums over the first variable must match the first margins."
    )
    sum_over_var2 = (
        df_raked.groupby(["var1", "var3", "draws"])
        .agg({"soln": "sum"})
        .reset_index()
        .merge(df_margins_2, on=["var1", "var3", "draws"])
    )
    assert np.allclose(sum_over_var2["soln"], sum_over_var2["value"], 1.0e-4), (
        "For the loop, the sums over the second variable must match the second margins."
    )
    sum_over_var3 = (
        df_raked.groupby(["var1", "var2", "draws"])
        .agg({"soln": "sum"})
        .reset_index()
        .merge(df_margins_3, on=["var1", "var2", "draws"])
    )
    assert np.allclose(sum_over_var3["soln"], sum_over_var3["value"], 1.0e-4), (
        "For the loop, the sums over the third variable must match the third margins."
    )

    # Use parallelization
    # The matrix in the objective of the dual has shape (60000, 36000) and full rank.
    # The solver converges correctly.
    data_builder = DataBuilderParallel(
        dim_specs={"var1": -1, "var2": -1, "var3": -1},
        dim_parallel=["draws"],
        value="value",
        weights="weights",
    )
    data = data_builder.build(df)
    solver = DualSolverParallel(distance="entropic", data=data)
    df_raked_parallel = solver.solve()

    # Check the results for the parallelization
    sum_over_var1 = (
        df_raked_parallel.groupby(["var2", "var3", "draws"])
        .agg({"soln": "sum"})
        .reset_index()
        .merge(df_margins_1, on=["var2", "var3", "draws"])
    )
    assert np.allclose(sum_over_var1["soln"], sum_over_var1["value"], 1.0e-3), (
        "For the parallelization, the sums over the first variable must match the first margins."
    )
    sum_over_var2 = (
        df_raked_parallel.groupby(["var1", "var3", "draws"])
        .agg({"soln": "sum"})
        .reset_index()
        .merge(df_margins_2, on=["var1", "var3", "draws"])
    )
    assert np.allclose(sum_over_var2["soln"], sum_over_var2["value"], 1.0e-3), (
        "For the parallelization, the sums over the second variable must match the second margins."
    )
    sum_over_var3 = (
        df_raked_parallel.groupby(["var1", "var2", "draws"])
        .agg({"soln": "sum"})
        .reset_index()
        .merge(df_margins_3, on=["var1", "var2", "draws"])
    )
    assert np.allclose(sum_over_var3["soln"], sum_over_var3["value"], 1.0e-3), (
        "For the parallelization, the sums over the third variable must match the third margins."
    )

    # Compare results
    df_both = pd.merge(
        df_raked,
        df_raked_parallel,
        on=["var1", "var2", "var3", "draws"],
        how="inner",
    )
    assert np.allclose(df_both["soln_x"], df_both["soln_y"], 1.0e-3), (
        "The two rakings must give the same results."
    )
