import numpy as np
import numpy.typing as npt
from scipy.sparse.linalg import LinearOperator

from raking.special import div0


class Diag(LinearOperator):
    def __init__(self, diag: npt.NDArray) -> None:
        self.diag = np.asarray(diag).ravel()
        super().__init__(
            dtype=self.diag.dtype,
            shape=(self.diag.size, self.diag.size),
        )

    def _matvec(self, x: npt.NDArray) -> npt.NDArray:
        return self.diag * np.asarray(x).ravel()

    def _rmatvec(self, x: npt.NDArray) -> npt.NDArray:
        return self.diag * np.asarray(x).ravel()

    def solve(self, x: npt.NDArray, **kwargs) -> npt.NDArray:
        x = np.asarray(x).ravel()
        return div0(x, self.diag)
