import numpy as np
import numpy.typing as npt
from scipy.sparse.linalg import LinearOperator, aslinearoperator

from raking.linops.typing import LinearOperatorLike


class IdBlockDiag(LinearOperator):
    def __init__(self, op: LinearOperatorLike, n_blocks: int) -> None:
        self.op = aslinearoperator(op)
        self.n_blocks = n_blocks
        super().__init__(
            dtype=op.dtype, shape=(n_blocks * op.shape[0], n_blocks * op.shape[1])
        )

    def _matvec(self, x: npt.NDArray) -> npt.NDArray:
        X = np.asarray(x).ravel().reshape(self.n_blocks, self.op.shape[1])
        return (X @ self.op.T).ravel()

    def _rmatvec(self, x: npt.NDArray) -> npt.NDArray:
        X = np.asarray(x).ravel().reshape(self.n_blocks, self.op.shape[0])
        return (X @ self.op).ravel()
