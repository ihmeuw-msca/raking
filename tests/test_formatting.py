import pytest
import numpy as np
import pandas as pd
from raking.compute_constraints import (
    constraints_1D,
    constraints_2D,
    constraints_3D,
)
from raking.formatting_methods import (
    format_data_1D,
    format_data_2D,
    format_data_3D,
)


def test_format_data_1D():
    # Generate balanced vector
    I = 3
    rng = np.random.default_rng(0)
    y = rng.uniform(low=2.0, high=3.0, size=I)
    s = np.sum(y)
    # Generate the data frames
    var1 = np.arange(0, I)
    df_obs = pd.DataFrame({"value": y, "var1": var1})
    df_margins = pd.DataFrame({"value_agg_over_var1": [s]})
    # Get the formatted data
    (y, s, I, q, l, h) = format_data_1D(df_obs, df_margins, "var1")
    # Generate the constraints
    (A, s) = constraints_1D(s, I)
    # Verify that the constraint A beta = s is respected
    assert np.allclose(
        np.matmul(A, y), s
    ), "For the format_data_1D function, the constraint A y = s is not respected."


def test_format_data_2D():
    # Generate balanced matrix
    I = 3
    J = 5
    rng = np.random.default_rng(0)
    y = rng.uniform(low=2.0, high=3.0, size=(I, J))
    s1 = np.sum(y, axis=0)
    s2 = np.sum(y, axis=1)
    y = y.flatten(order="F")
    # Generate the data frames
    var1 = np.tile(np.arange(0, I), J)
    var2 = np.repeat(np.arange(0, J), I)
    df_obs = pd.DataFrame({"value": y, "var1": var1, "var2": var2})
    df_margins_1 = pd.DataFrame(
        {"var2": np.arange(0, J), "value_agg_over_var1": s1}
    )
    df_margins_2 = pd.DataFrame(
        {"var1": np.arange(0, I), "value_agg_over_var2": s2}
    )
    # Get the formatted data
    (y, s1, s2, I, J, q, l, h) = format_data_2D(
        df_obs, df_margins_1, df_margins_2, ["var1", "var2"]
    )
    # Generate the constraints
    (A, s) = constraints_2D(s1, s2, I, J)
    # Verify that the constraint A beta = s is respected
    assert np.allclose(
        np.matmul(A, y), s
    ), "For the format_data_2D function, the constraint A y = s is not respected."


def test_format_data_3D():
    # Generate balanced array
    I = 3
    J = 4
    K = 5
    rng = np.random.default_rng(0)
    y = rng.uniform(low=2.0, high=3.0, size=(I, J, K))
    s1 = np.sum(y, axis=0)
    s2 = np.sum(y, axis=1)
    s3 = np.sum(y, axis=2)
    y = y.flatten(order="F")
    s1 = s1.flatten(order="F")
    s2 = s2.flatten(order="F")
    s3 = s3.flatten(order="F")
    # Generate the observation data frame
    var1 = np.tile(np.arange(0, I), J * K)
    var2 = np.tile(np.repeat(np.arange(0, J), I), K)
    var3 = np.repeat(np.arange(0, K), I * J)
    df_obs = pd.DataFrame(
        {"value": y, "var1": var1, "var2": var2, "var3": var3}
    )
    # Generate the first margins data frame
    var2 = np.tile(np.arange(0, J), K)
    var3 = np.repeat(np.arange(0, K), J)
    df_margins_1 = pd.DataFrame(
        {"var2": var2, "var3": var3, "value_agg_over_var1": s1}
    )
    # Generate the second margins data frame
    var1 = np.tile(np.arange(0, I), K)
    var3 = np.repeat(np.arange(0, K), I)
    df_margins_2 = pd.DataFrame(
        {"var1": var1, "var3": var3, "value_agg_over_var2": s2}
    )
    # Generate the third margins data frame
    var1 = np.tile(np.arange(0, I), J)
    var2 = np.repeat(np.arange(0, J), I)
    df_margins_3 = pd.DataFrame(
        {"var1": var1, "var2": var2, "value_agg_over_var3": s3}
    )
    # Get the formatted data
    (y, s1, s2, s3, I, J, K, q, l, h) = format_data_3D(
        df_obs,
        df_margins_1,
        df_margins_2,
        df_margins_3,
        ["var1", "var2", "var3"],
    )
    # Generate the constraints
    (A, s) = constraints_3D(s1, s2, s3, I, J, K)
    # Verify that the constraint A beta = s is respected
    assert np.allclose(
        np.matmul(A, y), s
    ), "For the format_data_3D function, the constraint A y = s is not respected."
