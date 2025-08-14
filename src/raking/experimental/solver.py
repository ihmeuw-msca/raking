import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.sparse as sps
from scipy.optimize import LinearConstraint, minimize

from raking.experimental.data import Data
from raking.experimental.distance import distance_map


class DualSolver:
    def __init__(self, distance: str, data: Data) -> None:
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
        return self.fun(self.mat_o @ x, order=0).sum() + self.vec_o @ x

    def gradient(self, x: npt.NDArray) -> npt.NDArray:
        return self.mat_o.T @ self.fun(self.mat_o @ x, order=1) + self.vec_o

    def hessian(self, x: npt.NDArray) -> sps.csc_matrix:
        d2 = self.fun(self.mat_o @ x, order=2)
        return (self.mat_o.T.multiply(d2)) @ self.mat_o

    def dual_to_primal(self, z: npt.NDArray) -> npt.NDArray:
        size_p = self.data["vec_p"].sum()

        vec_g = self.fun(self.mat_o @ z, order=1)
        x1 = vec_g[:size_p]

        vec_s = np.hstack([vec_g[size_p:], self.data["vec_b"]])
        vec_s = vec_s - self.data["mat_mc1"].dot(x1)
        x2 = self.data["mat_q"] @ (self.data["mat_mc2"].T @ vec_s)

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
        if x0 is None:
            x0 = np.zeros(self.mat_o.shape[1])

        constraints = None
        if self.vec_c.size > 0:
            constraints = [LinearConstraint(self.mat_c, self.vec_c, self.vec_c)]

        if method is None:
            method = "L-BFGS-B" if self.vec_c.size == 0 else "trust-constr"

        if options == None:
            if method == "L-BFGS-B":
                options={'ftol': 1.0e-11, 'gtol': 1.0e-11}
            if method == "trust-constr":
                options={'gtol': 1.0e-11, 'xtol': 1.0e-11}

        if method == "L-BFGS-B":
            self.result = minimize(
                self.objective,
                x0,
                method=method,
                jac=self.gradient,
                tol=tol,
                options=options
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
                options=options
            )
        
        soln = self.data["span"].copy()
        soln["soln"] = self.dual_to_primal(self.result.x)
        return soln


class PrimalSolver:
    def __init__(self, distance: str, data: Data) -> None:
        self.fun = distance_map[distance](
            y=data["vec_y"],
            w=data["vec_w"],
            l=data["vec_l"],
            u=data["vec_u"],
        ).fun

        self.data = data
        size_v, size_r = data["vec_p"].size, data["vec_p"].sum()

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

        self.mat_o = sps.csr_matrix(sps.vstack([mat_s, data["mat_m"]]))
        self.mat_c = data["mat_c"]
        self.vec_c = data["vec_b"]
        self.result = None

    def objective(self, x: npt.NDArray) -> float:
        return self.fun(self.mat_o @ x, order=0).sum()

    def gradient(self, x: npt.NDArray) -> npt.NDArray:
        return self.mat_o.T @ self.fun(self.mat_o @ x, order=1)

    def hessian(self, x: npt.NDArray) -> sps.csc_matrix:
        d2 = self.fun(self.mat_o @ x, order=2)
        return (self.mat_o.T.multiply(d2)) @ self.mat_o

    def solve(
        self,
        x0: npt.NDArray | None = None,
        method: str | None = None,
        options: dict | None = None,
    ) -> pd.DataFrame:
        if x0 is None:
            x0 = np.zeros(self.mat_o.shape[1])

        constraints = None
        if self.vec_c.size > 0:
            constraints = [LinearConstraint(self.mat_c, self.vec_c, self.vec_c)]

        if method is None:
            method = "L-BFGS-B" if self.vec_c.size == 0 else "trust-constr"

        if options == None:
            if method == "L-BFGS-B":
                options={'ftol': 1.0e-11, 'gtol': 1.0e-11}
            if method == "trust-constr":
                options={'gtol': 1.0e-11, 'xtol': 1.0e-11}

        if method == "L-BFGS-B":
            self.result = minimize(
                self.objective,
                x0,
                method=method,
                jac=self.gradient,
                tol=tol,
                options=options
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
                options=options
            )

        soln = self.data["span"].copy()
        soln["soln"] = self.result.x
        return soln
