"""Data classes."""

import itertools
from typing import TypedDict

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.sparse as sps
from pydantic import BaseModel

from raking.experimental.data import DataBuilder, _build_design_mat


class DataParallel(TypedDict):
    """Observations and constraints for the optimization problem.

    Parameters
    ----------
    N: int
        Parallelization dimension. Number of times that the problem needs to be repeated.
    vec_p : numpy.typing.NDArray
        Indicates whether observations that are not constraints nor margins are missing.
    vec_init: numpy.typing.NDArray
        Initial guess for the unknown raked values.
    vec_y : numpy.typing.NDArray
        Vector containing the values of the observations that are not constraints and not missing.
    vec_w : numpy.typing.NDArray
        Vector containing the weights corresponding to the observations in vec_y. Must be > 0 and < np.inf.
    vec_l : numpy.typing.NDArray
        Lower bounds for the observations that are not constraints (including aggregates).
    vec_u : numpy.typing.NDArray
        Upper bounds for the observations that are not constraints (including aggregates).
    vec_c_primal: numpy.typing.NDArray
        Contains the constraints for the primal problem formulation.
    vec_o_dual: numpy.typing.NDarray
        Used in the dot product with the unknown in the objective function for the dual problem formulation.
    vec_c_dual: numpy.typing.NDArray
        Contains the constraints for the dual problem formulation.
    vec_b : numpy.typing.NDArray
        Vector containing the values of the constraints.
    mat_o_primal: scipy.sparse.csc_matrix
        Used in the matrix product with the unknown in the objective function for the primal problem formulation.
    mat_c_primal: scipy.sparse.csc_matrix
        Matrix indicating how to sum the unknowns to get the constriants for the primal problem formulation.
    mat_o_dual: scipy.sparse.csc_matrix
        Used in the matrix product with the unknown in the objective function for the dual problem formulation.
    mat_c_dual: scipy.sparse.csc_matrix
        Matrix indicating how to sum the unknowns to get the constriants for the dual problem formulation.
    mat_mc1 : scipy.sparse.csr_matrix
        Matrix indicating how to sum the non-missing observations that are not constraints nor margins
        to get margins and constraints.
    mat_mc2 : scipy.sparse.csr_matrix
        Matrix indicating how to sum the missing observations that are not constraints nor margins
        to get margins and constraints.
    mat_q : numpy.typing.NDArray
        Matrix indicating how to get the missing observations once we know the raked margins and the constraints.
        Should be equal to [mat_mc2^T mat_mc2]-1
    span : pandas.DataFrame
        Contains the values taken by the categorical variables in the raking problem (excluding aggregates).
    """

    N: int
    vec_p: npt.NDArray
    vec_init: npt.NDArray
    vec_y: npt.NDArray
    vec_w: npt.NDArray
    vec_l: npt.NDArray | None
    vec_u: npt.NDArray | None
    vec_c_primal: npt.NDArray
    vec_o_dual: npt.NDArray
    vec_c_dual: npt.NDArray
    vec_b: npt.NDArray

    mat_o_primal: sps.csc_matrix
    mat_c_primal: sps.csc_matrix
    mat_o_dual: sps.csc_matrix
    mat_c_dual: sps.csc_matrix
    may_mc1: sps.csc_matrix
    mat_mc2: sps.csr_matrix
    mat_q: sps.csr_matrix

    span: pd.DataFrame


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
            df_loc = df_loc.loc[df_loc[dim] == dim0]
            df_loc = df_loc.drop(columns=[dim])
        data_builder = DataBuilder(
            dim_specs=self.dim_specs,
            value=self.value,
            weights=self.weights,
            bounds=self.bounds,
        )
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
        df_observ, df_constr = (
            df_loc.query("~is_constr"),
            df_loc.query("is_constr"),
        )
        df_observ = data_builder._expand_observ(df_observ)
        df_constr = data_builder._check_constr(df_constr)

        index = df_observ["is_margin"]
        vec_p_loc = (df_observ[~index][self.weights] > 0).to_numpy()
        columns = [name for name in data_builder.space.names]
        vec_init_loc = df_observ[~index][columns]
        vec_init_loc["vec_init"] = vec_p_loc
        mat_m = _build_design_mat(df_observ[index], data_builder.space)

        index = df_observ.eval(f"{self.weights} > 0")
        columns = [name for name in data_builder.space.names]
        vec_yw_loc = df_observ[index][columns]

        index = df_constr["included"]
        mat_c = _build_design_mat(df_constr[index], data_builder.space)
        vec_b_loc = df_constr[index][columns]

        mat_mc = sps.csc_matrix(sps.vstack([mat_m, mat_c]))
        mat_mc1_loc = sps.csc_matrix(mat_mc[:, vec_p_loc])
        mat_mc2_loc = sps.csc_matrix(mat_mc[:, ~vec_p_loc])
        mat_q_loc = data_builder._check_sufficiency(mat_mc2_loc)

        # Build the matrices and vectors for the primal problem
        size_v, size_r = vec_p_loc.size, vec_p_loc.sum()
        mat_s = sps.csr_matrix(
            (
                np.ones(size_r, dtype=int),
                (
                    np.arange(size_r, dtype=int),
                    np.arange(size_v, dtype=int)[vec_p_loc],
                ),
            ),
            shape=(size_r, size_v),
        )
        mat_o_primal_loc = sps.csr_matrix(sps.vstack([mat_s, mat_m]))
        mat_c_primal_loc = mat_c.copy()
        vec_c_primal_loc = vec_b_loc.copy()

        # Build the matrices and vectors for the dual problem
        size_m, size_c = mat_m.shape[0], mat_c.shape[0]
        mat_o_dual_loc = sps.csr_matrix(
            sps.vstack([-mat_mc1_loc.T, sps.eye(size_m, n=size_m + size_c)])
        )
        dict_vec_o_dual = {}
        for column in columns:
            dict_vec_o_dual[column] = np.repeat(np.nan, size_m)
        vec_o_dual_loc = pd.DataFrame(dict_vec_o_dual)
        vec_o_dual_loc = pd.concat([vec_o_dual_loc, vec_b_loc])
        mat_c_dual_loc = mat_mc2_loc.T
        vec_c_dual_loc = np.zeros(mat_c_dual_loc.shape[0])

        # Save the span of the first data set
        span_loc = data_builder.space.span().copy()

        # Extend the data builder to the entire problem
        # For the matrices, take the Kronecker product with identity matrix
        N = 1
        for dim in self.dim_parallel:
            N = N * len(df[dim].unique())
        data["N"] = N

        # Primal problem
        data["mat_o_primal"] = sps.kron(sps.eye(N, N), mat_o_primal_loc)
        data["mat_c_primal"] = sps.kron(sps.eye(N, N), mat_c_primal_loc)

        # Dual problem
        data["mat_o_dual"] = sps.kron(sps.eye(N, N), mat_o_dual_loc)
        data["mat_c_dual"] = sps.kron(sps.eye(N, N), mat_c_dual_loc)
        data["mat_mc1"] = sps.kron(sps.eye(N, N), mat_mc1_loc)
        data["mat_mc2"] = sps.kron(sps.eye(N, N), mat_mc2_loc)
        data["mat_q"] = sps.kron(sps.eye(N, N), mat_q_loc)

        # For logical vectors, concatenate
        data["vec_p"] = np.tile(vec_p_loc, N)

        # For the other vectors, keep the order of the rows
        dim_values = []
        for dim in self.dim_parallel:
            dim_values.append(df[dim].unique().tolist())
        parallel = pd.DataFrame(
            list(itertools.product(*dim_values)), columns=self.dim_parallel
        )

        # Both problems
        vec_init = pd.concat([vec_init_loc] * N)
        vec_yw = pd.concat([vec_yw_loc] * N)
        span = pd.concat([span_loc] * N)
        for dim in self.dim_parallel:
            vec_init[dim] = np.repeat(
                parallel[dim].to_numpy(), len(vec_init_loc)
            )
            vec_yw[dim] = np.repeat(parallel[dim].to_numpy(), len(vec_yw_loc))
            span[dim] = np.repeat(parallel[dim].to_numpy(), len(span_loc))
        vec_init = vec_init.merge(
            df, on=columns + self.dim_parallel, how="inner"
        ).fillna(0.0)
        vec_init[self.value] = vec_init[self.value].fillna(0.0)
        data["vec_init"] = vec_init[self.value] * vec_init["vec_init"]
        data["vec_y"] = vec_yw.merge(
            df, on=columns + self.dim_parallel, how="inner"
        )[self.value].to_numpy()
        data["vec_w"] = vec_yw.merge(
            df, on=columns + self.dim_parallel, how="inner"
        )[self.weights].to_numpy()
        data["vec_l"], data["vec_u"] = None, None
        if self.bounds is not None:
            lb, ub = self.bounds
            data["vec_l"] = vec_yw.merge(
                df, on=columns + self.dim_parallel, how="inner"
            )[lb].to_numpy()
            data["vec_u"] = vec_yw.merge(
                df, on=columns + self.dim_parallel, how="inner"
            )[ub].to_numpy()
        data["span"] = span

        # Primal problem
        vec_c_primal = pd.concat([vec_c_primal_loc] * N)
        for dim in self.dim_parallel:
            vec_c_primal[dim] = np.repeat(
                parallel[dim].to_numpy(), len(vec_c_primal_loc)
            )
        data["vec_c_primal"] = vec_c_primal.merge(
            df, on=columns + self.dim_parallel, how="inner"
        )[self.value].to_numpy()

        # Dual problem
        vec_o_dual = pd.concat([vec_o_dual_loc] * N)
        vec_b = pd.concat([vec_b_loc] * N)
        for dim in self.dim_parallel:
            vec_o_dual[dim] = np.repeat(
                parallel[dim].to_numpy(), len(vec_o_dual_loc)
            )
            vec_b[dim] = np.repeat(parallel[dim].to_numpy(), len(vec_b_loc))
        data["vec_o_dual"] = (
            vec_o_dual.merge(df, on=columns + self.dim_parallel, how="left")[
                self.value
            ]
            .fillna(0.0)
            .to_numpy()
        )
        data["vec_b"] = vec_b.merge(
            df, on=columns + self.dim_parallel, how="inner"
        )[self.value].to_numpy()
        data["vec_c_dual"] = np.tile(vec_c_dual_loc, N)

        return data
