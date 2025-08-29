"""Distance classes."""

from abc import ABC, abstractmethod
from typing import Callable, Literal

import numpy as np
import numpy.typing as npt

UFunction = Callable[[npt.NDArray], npt.NDArray]


class C2Function:
    def __init__(
        self, fun: UFunction, dfun: UFunction, d2fun: UFunction
    ) -> None:
        self.fun = fun
        self.dfun = dfun
        self.d2fun = d2fun

    def __call__(
        self, x: npt.NDArray, order: Literal[0, 1, 2] = 0
    ) -> npt.NDArray:
        match order:
            case 0:
                return self.fun(x)
            case 1:
                return self.dfun(x)
            case 2:
                return self.d2fun(x)
            case _:
                raise ValueError("Invalid order")


class Distance(ABC):
    """Distance between observations and raked values.

    Parameters
    ----------
    y : numpy.typing.NDArray
        Initial observations.
    w : numpy.typing.NDArray
        Weights (the bigger they are, the more certain the corresponding observation is).
    l : numpy.typing.NDArray
        Lower bounds for the observations and raked values.
    u : numpy.typing.NDArray
        Upper bound for the observations and raked values.
    fun : raking.experimental.distance.C2Function
        Distance function with its gradient and Hessian.
    conjugate_fun : raking.experimental.distance.C2Function
        Convex conjugate of the distance function with its gradient and Hessian.
    """

    def __init__(
        self,
        y: npt.NDArray,
        w: npt.NDArray,
        l: npt.NDArray | None = None,
        u: npt.NDArray | None = None,
    ) -> None:
        """Create Distance instance.

        Parameters
        ----------
        y : numpy.typing.NDArray
            Initial observations.
        w : numpy.typing.NDArray
            Weights (the bigger they are, the more certain the corresponding observation is).
        l : numpy.typing.NDArray
            Lower bounds for the observations and raked values.
        u : numpy.typing.NDArray
            Upper bound for the observations and raked values.

        Returns
        -------
        Distance
            Distance instance.
        """
        self.y, self.w, self.l, self.u = y, w, l, u

        self.fun = C2Function(
            lambda x: self.w * self._fun(x),
            lambda x: self.w * self._dfun(x),
            lambda x: self.w * self._d2fun(x),
        )
        self.conjugate_fun = C2Function(
            lambda z: self.w * self._conjugate_fun(z / self.w),
            lambda z: self._conjugate_dfun(z / self.w),
            lambda z: self._conjugate_d2fun(z / self.w) / self.w,
        )

    @abstractmethod
    def _fun(self, x: npt.NDArray) -> npt.NDArray:
        """Function definition"""

    @abstractmethod
    def _dfun(self, x: npt.NDArray) -> npt.NDArray:
        """Function derivative definition"""

    @abstractmethod
    def _d2fun(self, x: npt.NDArray) -> npt.NDArray:
        """Function second order derivative definition"""

    @abstractmethod
    def _conjugate_fun(self, z: npt.NDArray) -> npt.NDArray:
        """Function definition"""

    @abstractmethod
    def _conjugate_dfun(self, z: npt.NDArray) -> npt.NDArray:
        """Function derivative definition"""

    @abstractmethod
    def _conjugate_d2fun(self, z: npt.NDArray) -> npt.NDArray:
        """Function second order derivative definition"""


class EntropicDistance(Distance):
    """Entropic distance (preserve the sign of the observations)."""

    def _fun(self, x: npt.NDArray) -> npt.NDArray:
        return x * np.log(x / self.y) - (x - self.y)

    def _dfun(self, x: npt.NDArray) -> npt.NDArray:
        return np.log(x / self.y)

    def _d2fun(self, x: npt.NDArray) -> npt.NDArray:
        return 1.0 / x

    def _conjugate_fun(self, z: npt.NDArray) -> npt.NDArray:
        return self.y * np.exp(z)

    def _conjugate_dfun(self, z: npt.NDArray) -> npt.NDArray:
        return self.y * np.exp(z)

    def _conjugate_d2fun(self, z: npt.NDArray) -> npt.NDArray:
        return self.y * np.exp(z)


class Chi2Distance(Distance):
    """Chi2 distance (usually faster)."""

    def _fun(self, x: npt.NDArray) -> npt.NDArray:
        return 0.5 / self.y * (x - self.y) ** 2

    def _dfun(self, x: npt.NDArray) -> npt.NDArray:
        return x / self.y - 1.0

    def _d2fun(self, x: npt.NDArray) -> npt.NDArray:
        return 1.0 / self.y

    def _conjugate_fun(self, z: npt.NDArray) -> npt.NDArray:
        return self.y * (0.5 * z**2 + z)

    def _conjugate_dfun(self, z: npt.NDArray) -> npt.NDArray:
        return self.y * (z + 1.0)

    def _conjugate_d2fun(self, z: npt.NDArray) -> npt.NDArray:
        return self.y


class LogisticDistance(Distance):
    """Logistic distance (ensures that the raked values are bounded)."""

    def __init__(
        self,
        y: npt.NDArray,
        w: npt.NDArray,
        l: npt.NDArray | None = None,
        u: npt.NDArray | None = None,
    ) -> None:
        if l is None or u is None:
            raise ValueError("LogisticDistance requires bounds")
        super().__init__(y, w, l, u)

        self.n = self.u - self.l
        self.n_log_n = self.n * np.log(self.n)
        self.y_l = self.y - self.l
        self.y_u = self.y - self.u

    def _fun(self, x: npt.NDArray) -> npt.NDArray:
        x_l, x_u = x - self.l, x - self.u
        return x_l * np.log(x_l / self.y_l) - x_u * np.log(x_u / self.y_u)

    def _dfun(self, x: npt.NDArray) -> npt.NDArray:
        x_l, x_u = x - self.l, x - self.u
        return np.log(x_l / self.y_l) - np.log(x_u / self.y_u)

    def _d2fun(self, x: npt.NDArray) -> npt.NDArray:
        x_l, x_u = x - self.l, x - self.u
        return 1.0 / x_l - 1.0 / x_u

    def _conjugate_fun(self, z: npt.NDArray) -> npt.NDArray:
        r = np.zeros_like(z)
        pos = z >= 0
        neg = ~pos

        r[pos] = z[pos] * self.u[pos] + self.n[pos] * (
            np.log(self.y_l[pos] - self.y_u[pos] * np.exp(-z[pos]))
        )
        r[neg] = z[neg] * self.l[neg] + self.n[neg] * (
            np.log(self.y_l[neg] * np.exp(z[neg]) - self.y_u[neg])
        )
        r -= self.n_log_n
        return r

    def _conjugate_dfun(self, z: npt.NDArray) -> npt.NDArray:
        r = np.zeros_like(z)
        pos = z >= 0
        neg = ~pos

        r[pos] = self.l[pos] + self.n[pos] * self.y_l[pos] / (
            self.y_l[pos] - self.y_u[pos] * np.exp(-z[pos])
        )
        r[neg] = self.u[neg] + self.n[neg] * self.y_u[neg] / (
            self.y_l[neg] * np.exp(z[neg]) - self.y_u[neg]
        )
        return r

    def _conjugate_d2fun(self, z: npt.NDArray) -> npt.NDArray:
        r = np.zeros_like(z)
        pos = z >= 0
        neg = ~pos

        y_u = self.y_u[pos] * np.exp(-z[pos])
        r[pos] = -self.n[pos] * self.y_l[pos] * y_u / (self.y_l[pos] - y_u) ** 2
        y_l = self.y_l[neg] * np.exp(z[neg])
        r[neg] = -self.n[neg] * self.y_u[neg] * y_l / (y_l - self.y_u[neg]) ** 2
        return r


distance_map = {
    "entropic": EntropicDistance,
    "chi2": Chi2Distance,
    "logistic": LogisticDistance,
}
