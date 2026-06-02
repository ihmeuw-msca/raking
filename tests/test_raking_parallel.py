import pytest
import numpy as np
import pandas as pd
from raking.compute_constraints import (
    constraints_1D,
    constraints_1D_parallel,
    constraints_USHD_lower,
    constraints_USHD_lower_parallel,
)
from raking.raking_methods import raking_entropic, raking_entropic_parallel
from raking.run_raking import run_raking


def test_1D(example_1D_draws):
    df_obs = example_1D_draws.df_obs
    df_margins = example_1D_draws.df_margin

    # Loop on the draws
    draws = df_obs.draws.unique().tolist()
    df_raked = []
    for draw in draws:
        df_obs_loc = df_obs.loc[df_obs.draws == draw]
        df_margins_loc = df_margins.loc[df_margins.draws == draw]
        (df_obs_loc, Dphi_y, Dphi_s, sigma) = run_raking(
            dim=1,
            df_obs=df_obs_loc,
            df_margins=[df_margins_loc],
            var_names=["var1"],
            cov_mat=False,
        )
        df_raked.append(df_obs_loc)
    df_raked = pd.concat(df_raked)

    # Check the results for the loop on the draws
    sum_over_var1 = (
        df_raked.groupby(["draws"])
        .agg({"raked_value": "sum"})
        .reset_index()
        .merge(df_margins, on=["draws"])
    )
    assert np.allclose(
        sum_over_var1["raked_value"],
        sum_over_var1["value_agg_over_var1"],
        atol=1.0e-5,
    ), "The raked values do not sum to the margin."

    # Use parallelization
    df_raked_parallel = df_obs.copy(deep=True)
    df_raked_parallel = df_raked_parallel.sort_values(by=["draws", "var1"])
    df_margins_parallel = df_margins.copy(deep=True)
    df_margins_parallel = df_margins_parallel.sort_values(by=["draws"])

    I = len(df_raked_parallel.var1.unique())
    N = len(df_raked_parallel.draws.unique())

    s = df_margins_parallel.value_agg_over_var1.to_numpy()
    (A, s) = constraints_1D_parallel(s, I, N)
    y = df_raked_parallel["value"].to_numpy()
    q = np.ones(len(y))
    (beta, lambda_k, epsilon, iter_eps) = raking_entropic_parallel(y, A, s, q)
    df_raked_parallel["raked_value"] = beta

    # Check the results for the parallelization
    sum_over_var1 = (
        df_raked_parallel.groupby(["draws"])
        .agg({"raked_value": "sum"})
        .reset_index()
        .merge(df_margins, on=["draws"])
    )
    assert np.allclose(
        sum_over_var1["raked_value"],
        sum_over_var1["value_agg_over_var1"],
        atol=1.0e-5,
    ), "The raked values do not sum to the margin."

    # Compare results
    df_both = pd.merge(
        df_raked,
        df_raked_parallel,
        on=["var1", "draws"],
        how="inner",
    )
    assert np.allclose(
        df_both["raked_value_x"],
        df_both["raked_value_y"],
        atol=1.0e-11,
    ), "The two rakings must give the same results."


def test_USHD_lower(example_USHD_lower_draws):
    df_obs = example_USHD_lower_draws.df_obs
    df_margins_cause = example_USHD_lower_draws.df_margins_cause
    df_margins_county = example_USHD_lower_draws.df_margins_county
    df_margins_all_causes = example_USHD_lower_draws.df_margins_all_causes

    # Loop on the draws
    draws = df_obs.draws.unique().tolist()
    df_raked = []
    for draw in draws:
        df_obs_loc = df_obs.loc[df_obs.draws == draw]
        df_margins_cause_loc = df_margins_cause.loc[
            df_margins_cause.draws == draw
        ]
        df_margins_county_loc = df_margins_county.loc[
            df_margins_county.draws == draw
        ]
        df_margins_all_causes_loc = df_margins_all_causes.loc[
            df_margins_all_causes.draws == draw
        ]
        (df_obs_loc, Dphi_y, Dphi_s, sigma) = run_raking(
            dim="USHD_lower",
            df_obs=df_obs_loc,
            df_margins=[
                df_margins_cause_loc,
                df_margins_county_loc,
                df_margins_all_causes_loc,
            ],
            var_names=None,
            margin_names=["_inj", 1, 0],
            cov_mat=False,
            method="entropic",
        )
        df_raked.append(df_obs_loc)
    df_raked = pd.concat(df_raked)

    # Check the results for the loop on the draws
    sum_over_cause = (
        df_raked.loc[df_raked.race != 1]
        .groupby(["race", "county", "draws"], observed=True)
        .agg({"raked_value": "sum"})
        .reset_index()
        .merge(df_margins_all_causes, on=["race", "county", "draws"])
    )
    assert np.allclose(
        sum_over_cause["raked_value"],
        sum_over_cause["value_agg_over_cause"],
        atol=1.0e-4,
    ), "The sums over the cause must match the all causes deaths."
    sum_over_cause_race = (
        df_raked.loc[df_raked.race == 1]
        .groupby(["county", "draws"], observed=True)
        .agg({"raked_value": "sum"})
        .reset_index()
        .merge(df_margins_county, on=["county", "draws"])
    )
    assert np.allclose(
        sum_over_cause_race["raked_value"],
        sum_over_cause_race["value_agg_over_cause_race"],
        atol=1.0e-4,
    ), (
        "The sums over the cause and race must match the all causes all races deaths."
    )
    sum_over_race = (
        df_raked.loc[df_raked.race != 1]
        .groupby(["cause", "county", "draws"], observed=True)
        .agg({"raked_value": "sum"})
        .reset_index()
        .merge(
            df_raked.loc[df_raked.race == 1], on=["cause", "county", "draws"]
        )
    )
    assert np.allclose(
        sum_over_race["raked_value_x"],
        sum_over_race["raked_value_y"],
        atol=1.0e-4,
    ), "The sums over the race must match the all races deaths."
    sum_over_race_county = (
        df_raked.loc[df_raked.race != 1]
        .groupby(["cause", "draws"], observed=True)
        .agg({"raked_value": "sum"})
        .reset_index()
        .merge(df_margins_cause, on=["cause", "draws"])
    )
    assert np.allclose(
        sum_over_race_county["raked_value"],
        sum_over_race_county["value_agg_over_race_county"],
        atol=1.0e-4,
    ), "The sums over race and county must match the GBD values."

    # Use parallelization
    df_raked_parallel = df_obs.copy(deep=True)
    df_raked_parallel = df_raked_parallel.drop(columns=["upper"]).sort_values(
        by=["draws", "county", "race", "cause"]
    )
    df_margins_cause_parallel = df_margins_cause.copy(deep=True)
    df_margins_cause_parallel = df_margins_cause_parallel.sort_values(
        by=["draws", "cause"]
    )
    df_margins_county_parallel = df_margins_county.copy(deep=True)
    df_margins_county_parallel = df_margins_county_parallel.sort_values(
        by=["draws", "county"]
    )
    df_margins_all_causes_parallel = df_margins_all_causes.copy(deep=True)
    df_margins_all_causes = df_margins_all_causes.sort_values(
        by=["draws", "county", "race"]
    )

    I = len(df_raked_parallel.cause.unique())
    J = len(df_raked_parallel.race.unique()) - 1
    K = len(df_raked_parallel.county.unique())
    N = len(df_raked_parallel.draws.unique())

    s_cause = df_margins_cause_parallel.value_agg_over_race_county.to_numpy()
    s_county = np.reshape(
        df_margins_county_parallel.value_agg_over_cause_race.to_numpy(),
        (N, K),
        "C",
    )[:, 0 : (K - 1)].flatten("C")
    s_all_causes = (
        df_margins_all_causes_parallel.value_agg_over_cause.to_numpy()
    )
    (A, s) = constraints_USHD_lower_parallel(
        s_cause, s_county, s_all_causes, I, J, K, N
    )
    y = df_raked_parallel["value"].to_numpy()
    q = np.ones(len(y))
    (beta, lambda_k, epsilon, iter_eps) = raking_entropic_parallel(y, A, s, q)
    df_raked_parallel["raked_value"] = beta

    # Check the results for the parallelization
    sum_over_cause = (
        df_raked_parallel.loc[df_raked_parallel.race != 1]
        .groupby(["race", "county", "draws"], observed=True)
        .agg({"raked_value": "sum"})
        .reset_index()
        .merge(df_margins_all_causes_parallel, on=["race", "county", "draws"])
    )
    assert np.allclose(
        sum_over_cause["raked_value"],
        sum_over_cause["value_agg_over_cause"],
        atol=1.0e-4,
    ), "The sums over the cause must match the all causes deaths."
    sum_over_cause_race = (
        df_raked_parallel.loc[df_raked_parallel.race == 1]
        .groupby(["county", "draws"], observed=True)
        .agg({"raked_value": "sum"})
        .reset_index()
        .merge(df_margins_county_parallel, on=["county", "draws"])
    )
    assert np.allclose(
        sum_over_cause_race["raked_value"],
        sum_over_cause_race["value_agg_over_cause_race"],
        atol=1.0e-4,
    ), (
        "The sums over the cause and race must match the all causes all races deaths."
    )
    sum_over_race = (
        df_raked_parallel.loc[df_raked_parallel.race != 1]
        .groupby(["cause", "county", "draws"], observed=True)
        .agg({"raked_value": "sum"})
        .reset_index()
        .merge(
            df_raked_parallel.loc[df_raked_parallel.race == 1],
            on=["cause", "county", "draws"],
        )
    )
    assert np.allclose(
        sum_over_race["raked_value_x"],
        sum_over_race["raked_value_y"],
        atol=1.0e-4,
    ), "The sums over the race must match the all races deaths."
    sum_over_race_county = (
        df_raked_parallel.loc[df_raked_parallel.race != 1]
        .groupby(["cause", "draws"], observed=True)
        .agg({"raked_value": "sum"})
        .reset_index()
        .merge(df_margins_cause_parallel, on=["cause", "draws"])
    )
    assert np.allclose(
        sum_over_race_county["raked_value"],
        sum_over_race_county["value_agg_over_race_county"],
        atol=1.0e-4,
    ), "The sums over race and county must match the GBD values."

    # Compare results
    df_both = pd.merge(
        df_raked,
        df_raked_parallel,
        on=["cause", "race", "county", "draws"],
        how="inner",
    )
    assert np.allclose(
        df_both["raked_value_x"],
        df_both["raked_value_y"],
        atol=1.0e-10,
    ), "The two rakings must give the same results."
