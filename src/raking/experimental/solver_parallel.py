"""Solver classes."""

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.sparse as sps
from scipy.optimize import LinearConstraint, minimize
from scipy.sparse.linalg import LinearOperator

from raking.experimental.data_parallel import DataParallel
from raking.experimental.distance import distance_map
from raking.experimental.optim import NTResult, NTSolver
from raking.experimental.special import div0, log0


class HessianOperator(LinearOperator):
    """
    Transform the Hessian matrix into linear operator.
    """

    def __init__(self, hessian: npt.NDArray) -> None:
        self.hessian = hessian
        super().__init__(
            dtype=self.hessian.dtype,
            shape=self.hessian.shape,
        )

    def _matvec(self, x: npt.NDArray) -> npt.NDArray:
        return self.hessian @ x

    def _solve_direct(self, x: npt.NDArray, **kwargs) -> npt.NDArray:
        return sps.linalg.spsolve(self.hessian, x, **kwargs)

    def _solve_cg(self, x: npt.NDArray, **kwargs) -> npt.NDArray:
        soln, info = sps.linalg.cg(self.hessian, x, **kwargs)
        if info > 0:
            raise RuntimeError(
                f"CG solver didn't converge, with {info} iterations"
            )
        elif info < 0:
            raise RuntimeError(f"CG solver didn't converge: {info}")
        return soln

    def solve(
        self, x: npt.NDArray, solver_type: str = "direct", **kwargs
    ) -> npt.NDArray:
        if solver_type not in ["cg", "direct"]:
            raise ValueError(f"Unrecognized solver type: '{solver_type}'")
        if solver_type == "direct":
            return self._solve_direct(x, **kwargs)
        elif solver_type == "cg":
            return self._solve_cg(x, **kwargs)


class DualSolverParallel:
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
    result : scipy.optimize.OptimizeResult or NTResult
        Contains info on the solution and the convergence of the optimization problem.
    """

    def __init__(self, distance: str, data: DataParallel) -> None:
        """Create DualSolver instance.

        Parameters
        ----------
        distance : str
            Distance between observations and raked values.
            Currently, only chi2, entropic and logistic are implemented.
        data: raking.experimental.data_parallel.DataParallel
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
        self.mat_o = data["mat_o_dual"]
        self.vec_o = data["vec_o_dual"]
        self.mat_c = data["mat_c_dual"]
        self.vec_c = data["vec_c_dual"]
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

    def hessian(self, x: npt.NDArray) -> HessianOperator | sps.spmatrix:
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
        hessian_matrix = (self.mat_o.T.multiply(d2)) @ self.mat_o
        if self.vec_c.size == 0:
            return HessianOperator(hessian_matrix)
        else:
            return hessian_matrix

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
        N = self.data["N"]
        size_p = int(self.data["vec_p"].sum() / N)
        vec_g = self.fun(self.mat_o @ z, order=1)
        size_g = int(vec_g.size / N)

        x1 = np.ravel(np.reshape(vec_g, (size_g, N), "F")[:size_p, :], "F")
        size_b = int(self.data["vec_b"].size / N)
        vec_s = np.ravel(
            np.vstack(
                [
                    np.reshape(vec_g, (size_g, N), "F")[size_p:, :],
                    np.reshape(self.data["vec_b"], (size_b, N), "F"),
                ]
            ),
            "F",
        )
        vec_s = vec_s - self.data["mat_mc1"].dot(x1)
        x2 = self.data["mat_q"] @ (self.data["mat_mc2"].T @ vec_s)
        x = np.zeros_like(self.data["vec_p"], dtype=float)
        x[self.data["vec_p"]] = x1
        x[~self.data["vec_p"]] = x2
        return x

    def get_x0(self, x0: npt.NDArray | None) -> npt.NDArray:
        if x0 is None:
            # We only implement the initialization for entropic distance
            # as we have to deal with 0s in the observations and cannot use directly the distance map
            if self.distance != "entropic":
                x0 = np.zeros(self.mat_o.shape[1])
            else:
                # We need the matrix used in the objective of the primal problem
                # to transform initial guess for raked values into initial guess for the dual.
                mat_o = self.data["mat_o_primal"]
                # We compute the gradient of the entropic distance
                # while avoiding taking logarithms of 0s
                grad = log0(
                    div0(mat_o @ self.data["vec_init"], self.data["vec_y"])
                )
                # Taking the gradient of the Lagrangian, we have:
                # nabla f(zeta0,y) + (mat_m I \\ mat_c 0)^T lambda0
                x0 = sps.linalg.lsqr(self.mat_o, grad)[0]

        return x0

    def solve(
        self,
        x0: npt.NDArray | None = None,
        tol: float = 1.0e-11,
        options: dict | None = None,
        solver_options: dict | None = None,
    ) -> pd.DataFrame:
        """Solve the dual problem using scipy.optimize.minimize or the MSCA solver.

        Parameters
        ----------
        x0 : numpy.typing.NDArray
            Initial guess for the dual.
        tol: float
            Tolerance for termination. See scipy.optimize.minimize documentation for details.
        options : dict
            Additional parameters for the algorithm. See scipy.optimize.minimize documentation for details.
        solver_options: dict
            Chooses the solver for the linear system (mat_solve_method) and pass arguments to it (mat_solve_options).

        Returns
        -------
        soln : pandas.DataFrame
            Columns contain the categorical variables in the initial dataset (without the margins).
            The soln column contains the value of the raked observations.
        """
        # Initialization
        x0 = self.get_x0(x0)

        if self.vec_c.size == 0:
            if options is None:
                options = {}

            solver_options = solver_options or {}
            method = solver_options.get("mat_solve_method", "direct")
            solve_opts = solver_options.get("mat_solve_options", {})
            options = dict(options)
            options["mat_solve_method"] = method
            options["mat_solve_options"] = solve_opts

            interface = {
                "fun": self.objective,
                "grad": self.gradient,
                "hess": self.hessian,
            }
            optimizer = NTSolver(**interface)
            self.result = optimizer.minimize(x0=x0, **options)
        else:
            constraints = [LinearConstraint(self.mat_c, self.vec_c, self.vec_c)]
            if options == None:
                options = {"gtol": 1.0e-11, "xtol": 1.0e-11}
            self.result = minimize(
                self.objective,
                x0,
                method="trust-constr",
                jac=self.gradient,
                hess=self.hessian,
                constraints=constraints,
                tol=tol,
                options=options,
            )
        soln = self.data["span"].copy()
        soln["soln"] = self.dual_to_primal(self.result.x)
        return soln


class PrimalSolverParallel:
    """Solver for the primal problem.

    Parameters
    ----------
    data : raking.experimental.data_parallel.DataParallel
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

    def __init__(self, distance: str, data: DataParallel) -> None:
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
        self.mat_o = data["mat_o_primal"]
        self.mat_c = data["mat_c_primal"]
        self.vec_c = data["vec_c_primal"]
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
