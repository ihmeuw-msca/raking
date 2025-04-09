"""Script to profile the computation time of Niles' raking case.
See https://kernprof.readthedocs.io/en/latest/ for the documentation."""

import numpy as np
import pandas as pd

from line_profiler import profile

from raking.run_raking import run_raking

def generate_data(I, J):
    rng = np.random.default_rng(0)
    beta = rng.uniform(low=2.0, high=3.0, size=(I, J))
    s1 = np.sum(beta, axis=0)
    s2 = np.sum(beta, axis=1)
    y = beta + rng.normal(0.0, 0.1, size=beta.shape)
    # Observations data frame
    value = y.flatten(order='F')
    var1 = np.array([str(i) for i in np.tile(np.arange(1, I + 1), J)])
    var2 = np.array([str(i) for i in np.repeat(np.arange(1, J + 1), I)])
    df_obs = pd.DataFrame({'var1': var1, 'var2': var2, 'value': value})
    # First margins data frame
    var2 = np.array([str(i) for i in np.arange(1, J + 1)])
    df_margins_1 = pd.DataFrame({'var2': var2, 'value_agg_over_var1': s1})
    # Second margins data frame
    var1 = np.array([str(i) for i in np.arange(1, I + 1)])
    df_margins_2 = pd.DataFrame({'var1': var1, 'value_agg_over_var2': s2})
    return (df_obs, df_margins_1, df_margins_2)

@profile
def rake_chi2(df_obs, df_margins_1, df_margins_2):
    (df_raked, dummy1, dummy2, dumm3) = run_raking(
        dim=2,
        df_obs=df_obs,
        df_margins=[df_margins_1, df_margins_2],
        var_names=["var1", "var2"],
        cov_mat=False,
    )
    sum_over_var1 = (
        df_raked.groupby(["var2"])
        .agg({"raked_value": "sum"})
        .reset_index()
        .merge(df_margins_1, on="var2")
    )
    assert np.allclose(
        sum_over_var1["raked_value"], sum_over_var1["value_agg_over_var1"]
    ), "The sums over the first variable must match the first margins."
    sum_over_var2 = (
        df_raked.groupby(["var1"])
        .agg({"raked_value": "sum"})
        .reset_index()
        .merge(df_margins_2, on="var1")
    )
    assert np.allclose(
        sum_over_var2["raked_value"], sum_over_var2["value_agg_over_var2"]
    ), "The sums over the second variable must match the second margins."

@profile
def rake_entropic(df_obs, df_margins_1, df_margins_2):
    (df_raked, dummy1, dummy2, dumm3) = run_raking(
        dim=2,
        df_obs=df_obs,
        df_margins=[df_margins_1, df_margins_2],
        var_names=["var1", "var2"],
        method="entropic",
        cov_mat=False,
    )
    sum_over_var1 = (
        df_raked.groupby(["var2"])
        .agg({"raked_value": "sum"})
        .reset_index()
        .merge(df_margins_1, on="var2")
    )
    assert np.allclose(
        sum_over_var1["raked_value"], sum_over_var1["value_agg_over_var1"]
    ), "The sums over the first variable must match the first margins."
    sum_over_var2 = (
        df_raked.groupby(["var1"])
        .agg({"raked_value": "sum"})
        .reset_index()
        .merge(df_margins_2, on="var1")
    )
    assert np.allclose(
        sum_over_var2["raked_value"], sum_over_var2["value_agg_over_var2"]
    ), "The sums over the second variable must match the second margins."

def main():
    I = 36
    J = 1600
    (df_obs, df_margins_1, df_margins_2) = generate_data(I, J)
    rake_chi2(df_obs, df_margins_1, df_margins_2)
    rake_entropic(df_obs, df_margins_1, df_margins_2)

if __name__ == "__main__":
    main()

