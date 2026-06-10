"""Solver classes."""

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.sparse as sps
from scipy.optimize import LinearConstraint, minimize

from raking.experimental.data import Data
from raking.experimental.distance import distance_map


class DualSolver:
    """Solver for the dual problem.

    Parameters
    ----------
    data : raking.experimental.data.Data
        Contains observations data and constraints for the optimization problem.
    fun : raking.experimental.distance.Distance
        Convex conjugate of the distance between observations and raked values.
    mat_o : scipy.sparse.csr_matrix
    mat_c : scipy.sparse.csr_matrix
    vec_o : np.array
    vec_c : np.array
    result : scipy.optimize.OptimizeResult
        Contains info on the solution and the convergence of the optimization problem.
    """

    def __init__(self, distance: str, data: Data) -> None:
        """Create DualSolver instance.

        Parameters
        ----------
        distance : str
            Distance between observations and raked values.
            Currently, only chi2, entropic and logistic are implemented.
        data: raking.experimental.data.Data
            Contains observations data and constraints for the optimization problem.

        Returns
        -------
        DualSolver
            DualSolver instance.
        """
        self.distance = distance
        self.fun = distance_map[distance](
            y=data["vec_y"],
            w=data["vec_w"],
            l=data["vec_l"],
            u=data["vec_u"],
        ).conjugate_fun

        self.data = data
        size_m, size_c = data["mat_m"].shape[0], data["mat_c"].shape[0]
        self.mat_o = sps.csr_matrix(
            sps.vstack([-data["mat_mc1"].T, sps.eye(size_m, n=size_m + size_c)])
        )
        self.vec_o = np.hstack([np.zeros(size_m), data["vec_b"]])
        self.mat_c = data["mat_mc2"].T
        self.vec_c = np.zeros(self.mat_c.shape[0])
        self.result = None

    def objective(self, x: npt.NDArray) -> float:
        """Objective function for the dual optimization problem.

        Parameters
        ----------
        x : numpy.typing.NDArray
            Current value of the dual.

        Returns
        -------
        Objective function (float).
        """
        return self.fun(self.mat_o @ x, order=0).sum() + self.vec_o @ x

    def gradient(self, x: npt.NDArray) -> npt.NDArray:
        """Gradient of the objective function for the dual optimization problem.

        Parameters
        ----------
        x : numpy.typing.NDArray
            Current value of the dual.

        Returns
        -------
        Gradient of the objective function (numpy.typing.NDArray).
        """
        return self.mat_o.T @ self.fun(self.mat_o @ x, order=1) + self.vec_o

    def hessian(self, x: npt.NDArray) -> sps.csc_matrix:
        """Hessian of the objective function for the dual optimization problem.

        Parameters
        ----------
        x : numpy.typing.NDArray
            Current value of the dual.

        Returns
        -------
        Hessian of the objective function (scipy.sparse.csc_matrix).
        """
        d2 = self.fun(self.mat_o @ x, order=2)
        return (self.mat_o.T.multiply(d2)) @ self.mat_o

    def dual_to_primal(self, z: npt.NDArray) -> npt.NDArray:
        """Transforms dual solution into primal solution.

        Parameters
        ----------
        z : numpy.typing.NDArray
            Solution of the dual problem.

        Returns
        -------
        x : numpy.typing.NDArray
            Solution of the primal problem.
        """
        # size_p is the number of non-missing observations that are not margins and not constraints
        size_p = self.data["vec_p"].sum()

        # The inverse of the gradient of the distance function is equal to the gradient of the conjugate
        vec_g = self.fun(self.mat_o @ z, order=1)
        x1 = vec_g[:size_p]

        # vec_s contains the raked margins and the constraints
        vec_s = np.hstack([vec_g[size_p:], self.data["vec_b"]])
        # We remove from the raked margins and constraints the part computed with the raked non-missing observations.
        # vec_s now contains the part computed with the unknown raked missing observations
        vec_s = vec_s - self.data["mat_mc1"].dot(x1)
        # We now have vec_s = mat_mc2 x2 => x2 = [mat_mc2^T mat_mc2]-1 (mat_mc2^T vec_s)
        x2 = self.data["mat_q"] @ (self.data["mat_mc2"].T @ vec_s)

        # We assign the raked non-missing observations and the raked missing observations
        # in the output vector in the same order as the input data
        x = np.zeros_like(self.data["vec_p"], dtype=float)
        x[self.data["vec_p"]] = x1
        x[~self.data["vec_p"]] = x2
        return x

    def solve(
        self,
        x0: npt.NDArray | None = None,
        method: str | None = None,
        tol: float = 1.0e-11,
        options: dict | None = None,
    ) -> pd.DataFrame:
        """Solve the dual problem using scipy.optimize.minimize.

        Parameters
        ----------
        x0 : numpy.typing.NDArray
            Initial guess for the dual.
        method: str
            Optimization method. See scipy.optimize.minimize documentation for possible options.
        tol: float
            Tolerance for termination. See scipy.optimize.minimize documentation for details.
        options : dict
            Additional parameters for the algorithm. See scipy.optimize.minimize documentation for details.

        Returns
        -------
        soln : pandas.DataFrame
            Columns contain the categorical variables in the initial dataset (without the margins).
            The soln column contains the value of the raked observations.
        """
        if x0 is None:
            # We need the gradient of the initial function (not the conjugate) here
            fun = distance_map[self.distance](
                y=self.data["vec_y"],
                w=self.data["vec_w"],
                l=self.data["vec_l"],
                u=self.data["vec_u"],
            ).fun
            # We also need the matrix used in the objective of the primal problem
            size_v, size_r = self.data["vec_p"].size, self.data["vec_p"].sum()
            mat_s = sps.csr_matrix(
                (
                    np.ones(size_r, dtype=int),
                    (
                        np.arange(size_r, dtype=int),
                        np.arange(size_v, dtype=int)[self.data["vec_p"]],
                    ),
                ),
                shape=(size_r, size_v),
            )
            mat_o = sps.csr_matrix(sps.vstack([mat_s, self.data["mat_m"]]))
            grad = fun(mat_o @ self.data["vec_init"], order=1)
            x0 = sps.linalg.lsqr(self.mat_o, grad)[0]

        constraints = None
        if self.vec_c.size > 0:
            constraints = [LinearConstraint(self.mat_c, self.vec_c, self.vec_c)]

        if method is None:
            method = "L-BFGS-B" if self.vec_c.size == 0 else "trust-constr"

        if options == None:
            if method == "L-BFGS-B":
                options = {"ftol": 1.0e-11, "gtol": 1.0e-11}
            if method == "trust-constr":
                options = {"gtol": 1.0e-11, "xtol": 1.0e-11}

        if method == "L-BFGS-B":
            self.result = minimize(
                self.objective,
                x0,
                method=method,
                jac=self.gradient,
                tol=tol,
                options=options,
            )
        else:
            self.result = minimize(
                self.objective,
                x0,
                method=method,
                jac=self.gradient,
                hess=self.hessian,
                constraints=constraints,
                tol=tol,
                options=options,
            )

        soln = self.data["span"].copy()
        soln["soln"] = self.dual_to_primal(self.result.x)
        return soln


class PrimalSolver:
    """Solver for the primal problem.

    Parameters
    ----------
    data : raking.experimental.data.Data
        Contains observations data and constraints for the optimization problem.
    fun : raking.experimental.distance.Distance
        Distance between observations and raked values.
    mat_o : scipy.sparse.csr_matrix
        Matrix to transform the unknown raked values corresponding to missing and non-missing
        observations that are margins or non-margins, and compare them to the available observations.
    mat_c : scipy.sparse.csr_matrix
        Matrix to sum the unknown raked values corresponding to missing and non-missing
        observations that are margins or non-margins to get the constraints.
    vec_c : np.array
        Contains the constraints.
    result : scipy.optimize.OptimizeResult
        Contains info on the solution and the convergence of the optimization problem.
    """

    def __init__(self, distance: str, data: Data) -> None:
        """Create PrimalSolver instance.

        Parameters
        ----------
        distance : str
            Distance between observations and raked values.
            Currently, only chi2, entropic and logistic are implemented.
        data: raking.experimental.data.Data
            Contains observations data and constraints for the optimization problem.

        Returns
        -------
        PrimalSolver
            PrimalSolver instance.
        """
        self.fun = distance_map[distance](
            y=data["vec_y"],
            w=data["vec_w"],
            l=data["vec_l"],
            u=data["vec_u"],
        ).fun

        self.data = data
        # size_v is the number of observations (missing and non-missing) that are not margins and not constraints
        # size_r is the number of non-missing observations that are not margins and not constraints
        size_v, size_r = data["vec_p"].size, data["vec_p"].sum()

        # mat_s [missing + non-missing, non-constraints, non-margins obs] =
        # [non-missing, non-constraints, non-margins obs]
        mat_s = sps.csr_matrix(
            (
                np.ones(size_r, dtype=int),
                (
                    np.arange(size_r, dtype=int),
                    np.arange(size_v, dtype=int)[data["vec_p"]],
                ),
            ),
            shape=(size_r, size_v),
        )

        # mat_o is used to compute the distance between unknown raked values
        # and non-missing, non-constraints observations
        self.mat_o = sps.csr_matrix(sps.vstack([mat_s, data["mat_m"]]))
        # mat_c is used to take the missing and non-missing, non-margins, non constraints unknowned raked values
        # and transform them into the constraints that are stored in vec_c
        self.mat_c = data["mat_c"]
        self.vec_c = data["vec_b"]
        self.result = None

    def objective(self, x: npt.NDArray) -> float:
        """Objective function for the primal optimization problem.

        Parameters
        ----------
        x : numpy.typing.NDArray
            Current value of the raked observations.

        Returns
        -------
        Objective function (float).
        """
        return self.fun(self.mat_o @ x, order=0).sum()

    def gradient(self, x: npt.NDArray) -> npt.NDArray:
        """Gradient of the objective function for the primal optimization problem.

        Parameters
        ----------
        x : numpy.typing.NDArray
            Current value of the raked observations.

        Returns
        -------
        Gradient of the objective function (numpy.typing.NDArray).
        """
        return self.mat_o.T @ self.fun(self.mat_o @ x, order=1)

    def hessian(self, x: npt.NDArray) -> sps.csc_matrix:
        """Hessian of the objective function for the primal optimization problem.

        Parameters
        ----------
        x : numpy.typing.NDArray
            Current value of the raked observations.

        Returns
        -------
        Hessian of the objective function (scipy.sparse.csc_matrix).
        """
        d2 = self.fun(self.mat_o @ x, order=2)
        return (self.mat_o.T.multiply(d2)) @ self.mat_o

    def solve(
        self,
        x0: npt.NDArray | None = None,
        method: str | None = None,
        tol: float = 1.0e-11,
        options: dict | None = None,
    ) -> pd.DataFrame:
        """Solve the primal problem using scipy.optimize.minimize.

        Parameters
        ----------
        x0 : numpy.typing.NDArray
            Initial guess for the raked observations.
        method: str
            Optimization method. See scipy.optimize.minimize documentation for possible options.
        tol: float
            Tolerance for termination. See scipy.optimize.minimize documentation for details.
        options : dict
            Additional parameters for the algorithm. See scipy.optimize.minimize documentation for details.

        Returns
        -------
        soln : pandas.DataFrame
            Columns contain the categorical variables in the initial dataset (without the margins).
            The soln column contains the value of the raked observations.
        """
        if x0 is None:
            x0 = self.data["vec_init"]

        constraints = None
        if self.vec_c.size > 0:
            constraints = [LinearConstraint(self.mat_c, self.vec_c, self.vec_c)]

        if method is None:
            method = "L-BFGS-B" if self.vec_c.size == 0 else "trust-constr"

        if options == None:
            if method == "L-BFGS-B":
                options = {"ftol": 1.0e-11, "gtol": 1.0e-11}
            if method == "trust-constr":
                options = {"gtol": 1.0e-11, "xtol": 1.0e-11}

        if method == "L-BFGS-B":
            self.result = minimize(
                self.objective,
                x0,
                method=method,
                jac=self.gradient,
                tol=tol,
                options=options,
            )
        else:
            self.result = minimize(
                self.objective,
                x0,
                method=method,
                jac=self.gradient,
                hess=self.hessian,
                constraints=constraints,
                tol=tol,
                options=options,
            )

        soln = self.data["span"].copy()
        soln["soln"] = self.result.x
        return soln
