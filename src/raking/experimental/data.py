"""Data classes."""

import itertools
import operator
from typing import TypedDict

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.sparse as sps
from pydantic import BaseModel

from raking.experimental.dimension import Dimension, Space

# pd.set_option("future.no_silent_downcasting", True)


class Data(TypedDict):
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
    span : pandas.DataFrame
        Contains the values taken by the categorical variables in the raking problem (excluding aggregates).
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

    span: pd.DataFrame


class DataBuilder(BaseModel):
    """Specify observations and constraints for the optimization problem.

    Parameters
    ----------
    dim_specs : dict
        Keys = Categorical variables. Values = Code corresponding to the aggregate all categories.
        Example: If we rake each cause to all causes (encoded by -1): dim_specs={'cause': -1}
    value : str
        Name of the column containing the initial observations in the initial data frame.
    weights : str
        Name of the column containing the weights in the initial data frame.
    bounds : tuple[str, str]
        Names of the columns containing the lower and upper bounds (if using logistic distance).
    space : raking.experimental.dimension.Space
    """

    dim_specs: dict[str, int | str]
    value: str
    weights: str
    bounds: tuple[str, str] | None = None
    space: Space | None = None

    def build(self, df: pd.DataFrame) -> Data:
        """Build the observations and constraints for the optimization problem.

        Parameters
        ----------
        df : pandas DataFrame
            Contains one column for each of the keys in self.dim_specs,
            one column for the 'value', one column for the 'weights'
            and optionally two columns for the 'bounds'.

        Returns
        -------
        data : raking.experimental.data.Data
            Contains observations data and constraints for the optimization problem.
        """
        data = {}
        self._build_space(df)
        df = (
            df.pipe(self._subset_columns)
            .pipe(self._check_duplication)
            .pipe(self._check_weights)
            .pipe(self._check_value)
            .pipe(self._assign_level)
            .pipe(self._assign_indicators)
            .pipe(self._sort_rows)
        )
        df_observ, df_constr = df.query("~is_constr"), df.query("is_constr")
        df_observ = self._expand_observ(df_observ)
        df_constr = self._check_constr(df_constr)

        index = df_observ["is_margin"]
        data["vec_p"] = (df_observ[~index][self.weights] > 0).to_numpy()
        data["mat_m"] = _build_design_mat(df_observ[index], self.space)

        index = df_observ.eval(f"{self.weights} > 0")
        data["vec_y"] = df_observ[index][self.value].to_numpy()
        data["vec_w"] = df_observ[index][self.weights].to_numpy()

        index = df_constr["included"]
        data["mat_c"] = _build_design_mat(df_constr[index], self.space)
        data["vec_b"] = df_constr[index][self.value].to_numpy()

        data["vec_l"], data["vec_u"] = None, None
        if self.bounds is not None:
            data["vec_l"], data["vec_u"] = self._check_bounds(
                df_observ,
                data["mat_m"],
                data["vec_y"],
                data["mat_c"],
                data["vec_b"],
            )

        mat_mc = sps.csc_matrix(sps.vstack([data["mat_m"], data["mat_c"]]))
        data["mat_mc1"] = sps.csc_matrix(mat_mc[:, data["vec_p"]])
        data["mat_mc2"] = sps.csc_matrix(mat_mc[:, ~data["vec_p"]])
        data["mat_q"] = self._check_sufficiency(data["mat_mc2"])

        data["span"] = self.space.span().copy()

        return data

    def _build_space(self, df: pd.DataFrame) -> None:
        """Look for categorical variables and create the raking space."""
        dims = [
            Dimension.from_pandas_series(df[name], null)
            for name, null in self.dim_specs.items()
        ]
        dims.sort(key=operator.attrgetter("size"))
        self.space = Space(dimensions=dims)

    def _subset_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only the columns in the data frame that are used for the raking."""
        columns = list(self.space.names) + [self.value, self.weights]
        if self.bounds is not None:
            columns.extend(self.bounds)
        return df[columns].copy()

    def _check_duplication(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check for duplicated rows in the initial observations data frame."""
        if df.duplicated(subset=self.space.names).any():
            raise ValueError("There are duplicated observations or constraints")
        return df

    def _check_weights(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check for negative weights in the initial observations data frame."""
        column = self.weights
        if (df[column] < 0).any():
            raise ValueError(f"Column '{column}' contains negative values")
        return df

    def _check_value(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check for missing values in the initial observations data frame."""
        if df.query(f"{self.weights} > 0")[self.value].isna().any():
            raise ValueError(f"Column '{self.value}' contains missing values")
        return df

    def _assign_level(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check if an observation is an aggregate and how many categorical variables are aggregated."""
        df["level"] = df.eval(" + ".join(("0", *self.space.isnull)))
        return df

    def _assign_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check if an observation is a constraint and/or a margin."""
        df["is_constr"] = np.isinf(df[self.weights])
        df["is_margin"] = df.eval(" | ".join(self.space.isnull))
        return df

    def _sort_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort the rows of the data frame by constraint status, margin status and categorical variable."""
        columns = [f"{name}_order" for name in self.space.names]
        for column, dim in zip(columns, self.space.dimensions):
            df[column] = df[dim.name].map(dim.order)
        df = df.sort_values(["is_constr", "level"] + columns, ignore_index=True)
        df = df.drop(columns=columns)
        return df

    def _expand_observ(self, df: pd.DataFrame) -> pd.DataFrame:
        """Expand the observations to the missing values (that are not aggregates)."""
        df = (
            self.space.span()
            .copy()
            .assign(is_constr=False, is_margin=False)
            .merge(df, how="outer")
        )
        df[self.weights] = df[self.weights].fillna(0.0)
        df = self._sort_rows(df)
        return df.reset_index(drop=True)

    def _check_bounds(
        self,
        df_observ: pd.DataFrame,
        mat_m: sps.csc_matrix,
        vec_y: npt.NDArray,
        mat_c: sps.csc_matrix,
        vec_b: npt.NDArray,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Check if the given observations are within the bounds."""
        lb, ub = self.bounds
        df_observ[lb] = df_observ[lb].fillna(-np.inf)
        df_observ[ub] = df_observ[ub].fillna(np.inf)

        if df_observ.eval(f"{lb} > {ub}").any():
            raise ValueError("Lower bounds are greater than upper bounds")

        index = df_observ["is_margin"]
        vec_ls = df_observ[~index][lb]
        vec_us = df_observ[~index][ub]

        vec_lc_infer = mat_c.dot(vec_ls)
        vec_uc_infer = mat_c.dot(vec_us)
        vec_lm_infer = mat_m.dot(vec_ls)
        vec_um_infer = mat_m.dot(vec_us)

        vec_lm = df_observ.query("is_margin")[lb]
        vec_um = df_observ.query("is_margin")[ub]

        if (vec_b < vec_lc_infer).any() or (vec_b > vec_uc_infer).any():
            raise ValueError("Bounds and constraints are inconsistent")

        if (vec_lm > vec_um_infer).any() or (vec_um < vec_lm_infer).any():
            raise ValueError("Bounds for margin observations are inconsistent")

        df_observ.loc[index, lb] = np.maximum(vec_lm, vec_lm_infer)
        df_observ.loc[index, ub] = np.minimum(vec_um, vec_um_infer)

        vec_l = df_observ[lb].to_numpy()
        vec_u = df_observ[ub].to_numpy()

        if (vec_y < vec_l).any() or (vec_y > vec_u).any():
            raise ValueError("There are infeasible observations")

        return vec_l, vec_u

    def _check_constr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check if the constraints are consistent."""
        df = df.copy()
        df["index"] = df.index
        df["source"] = [(i,) for i in df["index"]]
        df["included"] = True
        df_by_level = {
            i: df.query(f"level == {i}").reset_index(drop=True)
            for i in range(self.space.ndim + 1)
        }

        for i, j in itertools.combinations(range(self.space.ndim + 1), 2):
            df_i = df_by_level[i]
            if df_i.empty:
                continue
            df_i, df_i_to_j = _resolve_same_level_duplicity(
                df_i, j, self.space, self.value
            )
            if df_i_to_j.empty:
                continue
            df_j = _resolve_upper_level_duplicity(
                df_i_to_j, df_by_level[j], j, self.space, self.value
            )
            df_by_level[i] = df_i
            df_by_level[j] = df_j

        df = pd.concat(df_by_level.values(), axis=0, ignore_index=True)
        df = df.query("index != -1").reset_index(drop=True)
        return df

    def _check_sufficiency(self, mat: sps.csc_matrix) -> npt.NDArray:
        """Check if we can extract the missing values form the other information."""
        _, mat_q, _, _ = _extract_independent_rows(mat.T)
        if mat_q.shape[0] < mat.shape[1]:
            raise ValueError(
                "Not enough information to solve the raking problem"
            )
        return mat_q


def _resolve_same_level_duplicity(
    df_i: pd.DataFrame, j: int, space: Space, value: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    i = int(df_i["level"].iloc[0])
    df_i_to_j = pd.concat(
        [
            _agg_constr(df_i, *spaces, value)
            for spaces in space.split((j - i, i))
        ],
        axis=0,
    )
    index = df_i_to_j.duplicated(subset=space.names)
    if index.any():
        df_dup, df_i_to_j = df_i_to_j[index], df_i_to_j[~index]
        df_cmp = df_i_to_j.merge(
            df_dup[list(space.names) + [value]],
            on=space.names,
            suffixes=("", "_alt"),
        )
        df_check = df_cmp.query(f"{value}.notna() & {value}_alt.notna()")
        if not np.allclose(df_check[value], df_check[f"{value}_alt"]):
            raise ValueError("Constraints are not consistent")

        index = df_i["index"].isin(df_dup["source"].map(max))
        df_i.loc[index, "included"] = False
    return df_i, df_i_to_j


def _resolve_upper_level_duplicity(
    df_i_to_j: pd.DataFrame,
    df_j: pd.DataFrame,
    j: int,
    space: Space,
    value: str,
) -> pd.DataFrame:
    df_cmp = df_j.merge(
        df_i_to_j,
        on=space.names,
        how="outer",
        suffixes=("", "_alt"),
    )
    df_cmp["included"] = df_cmp["included"].fillna(False).astype(bool)
    df_cmp["level"] = df_cmp["level"].fillna(j).astype(int)
    # check if values are aligned
    df_sub = df_cmp.query(f"{value}.notna() & {value}_alt.notna()")
    if not np.allclose(df_sub[value], df_sub[f"{value}_alt"]):
        raise ValueError("Constraints are not consistent")

    if df_cmp.eval(
        f"{value}.notna() & {value}_alt.notna() & (~included)"
    ).any():
        raise ValueError("Something is wrong!")

    index = df_cmp.eval(f"{value}_alt.notna()")
    df_cmp.loc[index, "included"] = False
    df_cmp.loc[index, "source"] = df_cmp.loc[index, "source_alt"]
    df_cmp["index"] = df_cmp["index"].fillna(-1).astype(int)
    return df_cmp[df_j.columns].copy()


def _agg_constr(
    df: pd.DataFrame, toagg: Space, agged: Space, others: Space, value: str
) -> pd.DataFrame:
    by = list(others.names)
    df_work = df.query(" & ".join((*toagg.notnull, *agged.isnull)))
    if len(by) > 0:
        df_sel = df_work.groupby(by).agg({"included": ["sum", "count"]})
        df_sel.columns = ["num_included", "size"]
        df_sel = df_sel.query(
            f"(num_included > 0) & (size == {toagg.size})"
        ).reset_index()
        df_work = df_work.merge(df_sel[by])
        df_work = df_work.groupby(by).agg(
            {
                value: "sum",
                "source": lambda x: tuple(itertools.chain.from_iterable(x)),
            }
        )
        df_work.columns = [value, "source"]
        df_work = df_work.reset_index()
    else:
        num_included, size = df_work["included"].sum(), len(df_work)
        if num_included > 0 and size == toagg.size:
            source = tuple(itertools.chain.from_iterable(df_work["source"]))
            df_work = pd.DataFrame(
                {value: df_work[value].sum(), "source": [source]}, index=[0]
            )
            for dim in agged.dimensions:
                df_work[dim.name] = dim.null
        else:
            df_work = pd.DataFrame(
                columns=list(agged.names) + [value, "source"]
            )
    for dim in itertools.chain(toagg.dimensions, agged.dimensions):
        df_work[dim.name] = dim.null
    return df_work


def _build_design_mat(df: pd.DataFrame, space: Space) -> sps.csc_matrix:
    """Returns a matrix indicating how to sum the observations to get the aggregates."""
    df = df.reset_index(drop=True)
    mat_shape = (len(df), space.size)

    row_indices = np.empty(shape=(0,), dtype=int)
    col_indices = np.empty(shape=(0,), dtype=int)

    for i, row in df.iterrows():
        col_index = np.asarray(
            space.index(tuple(row[name] for name in space.names))
        )
        row_index = np.asarray([i] * col_index.size)

        row_indices = np.hstack([row_indices, row_index])
        col_indices = np.hstack([col_indices, col_index])

    val = np.ones_like(row_indices, dtype=int)
    return sps.csc_matrix((val, (row_indices, col_indices)), shape=mat_shape)


def _extract_independent_rows(
    mat: sps.csr_matrix | sps.csc_matrix | sps.coo_matrix,
) -> tuple[sps.csr_matrix, npt.NDArray, sps.csr_matrix, npt.NDArray]:
    mat = sps.csr_matrix(mat)
    mat_r = sps.csr_matrix((0, mat.shape[1]), dtype=mat.dtype)
    mat_m = np.empty(shape=(0, 0), dtype=float)

    a_len = 0
    a_idx = np.empty(shape=(0,), dtype=bool)
    a_row = np.empty(shape=(0,), dtype=int)
    a_col = np.empty(shape=(0,), dtype=int)
    a_val = np.empty(shape=(0,), dtype=float)

    for i in range(mat.shape[0]):
        new_r = mat[i]
        y = mat_r.dot(new_r.T).toarray().ravel()
        a = mat_m.dot(y)
        b = (new_r.data**2).sum() - y.dot(a)
        a_idx = np.append(a_idx, np.isclose(b, 0.0))
        if a_idx[-1]:
            a_len += 1
            a_row = np.hstack([a_row, np.repeat(a_len - 1, a.size)])
            a_col = np.hstack([a_col, np.arange(a.size, dtype=int)])
            a_val = np.hstack([a_val, a])
        else:
            c = a / b
            mat_r = sps.vstack([mat_r, new_r])
            mat_m = np.block(
                [
                    [mat_m + np.outer(c, a), -c.reshape(-1, 1)],
                    [-c.reshape(1, -1), 1.0 / b],
                ]
            )

    mat_a = sps.csr_matrix(
        (a_val, (a_row, a_col)), shape=(a_len, mat_r.shape[0])
    )
    return mat_r, mat_m, mat_a, a_idx
