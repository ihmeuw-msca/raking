"""Data classes."""

import itertools
import operator
from typing import TypedDict

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.sparse as sps
from pandas.api.types import CategoricalDtype
from pydantic import BaseModel

from raking.experimental.data import Data, DataBuilder, _build_design_mat
from raking.experimental.dimension import Dimension, Space


class DataParallel(TypedDict):
    """Observations and constraints for the optimization problem.

    Parameters
    ----------
    vec_p : numpy.typing.NDArray
        Indicates whether observations that are not constraints nor margins are missing.
    vec_y : numpy.typing.NDArray
        Vector containing the values of the observations that are not constraints and not missing.
    vec_w : numpy.typing.NDArray
        Vector containing the weights corresponding to the observations in vec_y. Must be > 0 and < np.inf.
    vec_b : numpy.typing.NDArray
        Vector containing the values of the constraints.
    vec_l : numpy.typing.NDArray
        Lower bounds for the observations that are not constraints (including aggregates).
    vec_u : numpy.typing.NDArray
        Upper bounds for the observations that are not constraints (including aggregates).
    mat_m : scipy.sparse.csc_matrix
        Matrix indicating how to sum the observations to get the margins that are not constraints.
    mat_c : scipy.sparse.csc_matrix
        Matrix indicating how to sum the observations to get the constraints.
    mat_mc1 : scipy.sparse.csr_matrix
        Matrix indicating how to sum the observations that are not missing to get margins and constraints.
    mat_mc2 : scipy.sparse.csr_matrix
        Matrix indicating how to sum the observations that are missing to get margins and constraints.
    mat_q : numpy.typing.NDArray
    """

    vec_p: npt.NDArray
    vec_y: npt.NDArray
    vec_w: npt.NDArray
    vec_b: npt.NDArray
    vec_l: npt.NDArray | None
    vec_u: npt.NDArray | None
    mat_m: sps.csc_matrix
    mat_c: sps.csc_matrix

    mat_mc1: sps.csc_matrix
    mat_mc2: sps.csr_matrix
    mat_q: npt.NDArray

class DataBuilderParallel(BaseModel):
    """Specify observations and constraints for the optimization problem.

    Parameters
    ----------
    dim_specs : dict
        Keys = Categorical variables. Values = Code corresponding to the aggregate all categories.
        Example: If we rake each cause to all causes (encoded by -1): dim_specs={'cause': -1}
    dim_parallel : list
        List of categorical variables over which we want to loop the raking process.
        Example: draws, years, age groups.
    value : str
        Name of the column containing the initial observations in the initial data frame.
    weights : str
        Name of the column containing the weights in the initial data frame.
    bounds : tuple[str, str]
        Names of the columns containing the lower and upper bounds (if using logistic distance).
    """

    dim_specs: dict[str, int | str]
    dim_parallel: list
    value: str
    weights: str
    bounds: tuple[str, str] | None = None

    def build(self, df: pd.DataFrame) -> DataParallel:
        """Build the observations and constraints for the optimization problem.

        Parameters
        ----------
        df : pandas DataFrame
            Contains one column for each of the keys in self.dim_specs and self.dim_parallel,
            one column for the 'value', one column for the 'weights'
            and optionally two columns for the 'bounds'.

        Returns
        -------
        data : raking.experimental.data_parallel.DataParallel
            Contains observations data and constraints for the optimization problem.
        """
        data = {}

        # Take the first data set and build the corresponding single data builder
        df_loc = df.copy(deep=True)
        for dim in self.dim_parallel:
            dim0 = df[dim].unique().tolist()[0]
            df_loc = df_loc.loc[df_loc[dim]==dim0]
            df_loc = df_loc.drop(columns=[dim])
        data_builder = DataBuilder(dim_specs=self.dim_specs, value=self.value, weights=self.weights, bounds=self.bounds)
        data_builder._build_space(df_loc)
        df_loc = (
            df_loc.pipe(data_builder._subset_columns)
                  .pipe(data_builder._check_duplication)
                  .pipe(data_builder._check_weights)
                  .pipe(data_builder._check_value)
                  .pipe(data_builder._assign_level)
                  .pipe(data_builder._assign_indicators)
                  .pipe(data_builder._sort_rows)
        )
        df_observ, df_constr = df_loc.query("~is_constr"), df_loc.query("is_constr")
        df_observ = data_builder._expand_observ(df_observ)
        df_constr = data_builder._check_constr(df_constr)

        index = df_observ["is_margin"]
        vec_p_loc = (df_observ[~index][self.weights] > 0).to_numpy()
        mat_m_loc = _build_design_mat(df_observ[index], data_builder.space)

        index = df_observ.eval(f"{self.weights} > 0")
        columns = [name for name in data_builder.space.names]
        vec_yw_loc = df_observ[index][columns]

        index = df_constr["included"]
        mat_c_loc = _build_design_mat(df_constr[index], data_builder.space)
        vec_b_loc = df_constr[index][columns]

        mat_mc_loc = sps.csc_matrix(sps.vstack([mat_m_loc, mat_c_loc]))
        mat_mc1_loc = sps.csc_matrix(mat_mc_loc[:, vec_p_loc])
        mat_mc2_loc = sps.csc_matrix(mat_mc_loc[:, ~vec_p_loc])
        mat_q_loc = data_builder._check_sufficiency(mat_mc2_loc)

        span_loc = data_builder.space.span().copy()

        # For the matrices, take Kronecker product with identity matrix
        N = 1
        for dim in self.dim_parallel:
            N = N * len(df[dim].unique())
        data["mat_m"] = sps.kron(sps.eye(N, N), mat_m_loc)
        data["mat_c"] = sps.kron(sps.eye(N, N), mat_c_loc)
        data["mat_mc1"] = sps.kron(sps.eye(N, N), mat_mc1_loc)
        data["mat_mc2"] = sps.kron(sps.eye(N, N), mat_mc2_loc)
        data["mat_q"] = sps.kron(sps.eye(N, N), mat_q_loc)

        # For logical vectors, concatenate
        data["vec_p"] = np.tile(vec_p_loc, N)

        # For the other vectors, keep the order of the rows
        vec_yw = pd.concat([vec_yw_loc]*N)
        vec_b = pd.concat([vec_b_loc]*N)
        span = pd.concat([span_loc]*N)
        dim_values = []
        for dim in self.dim_parallel:
            dim_values.append(df[dim].unique().tolist())
        parallel = pd.DataFrame(list(itertools.product(*dim_values)), columns=self.dim_parallel)
        for dim in self.dim_parallel:
            vec_yw[dim] = np.repeat(parallel[dim].to_numpy(), len(vec_yw_loc))
            vec_b[dim] = np.repeat(parallel[dim].to_numpy(), len(vec_b_loc))
            span[dim] = np.repeat(parallel[dim].to_numpy(), len(span_loc))
        data["vec_y"] = vec_yw.merge(df, on=columns + self.dim_parallel, how="inner")[self.value].to_numpy()
        data["vec_w"] = vec_yw.merge(df, on=columns + self.dim_parallel, how="inner")[self.weights].to_numpy()
        data["vec_b"] = vec_b.merge(df, on=columns + self.dim_parallel, how="inner")[self.value].to_numpy()
        data["vec_l"], data["vec_u"] = None, None
        if self.bounds is not None:
            lb, ub = self.bounds
            data["vec_l"] = vec_yw.merge(df, on=columns + self.dim_parallel, how="inner")[lb].to_numpy()
            data["vec_u"] = vec_yw.merge(df, on=columns + self.dim_parallel, how="inner")[ub].to_numpy()
        data["span"] = span
        
        return data

