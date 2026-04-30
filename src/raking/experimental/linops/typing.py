from typing import Final, Protocol, Self, TypeAlias

import numpy as np
from scipy.sparse.linalg import LinearOperator


class Matrix(Protocol):
    ndim: Final[int] = 2
    shape: tuple[int, int]
    dtype: np.dtype

    def __matmul__(self, x: np.ndarray | Self) -> np.ndarray | Self:
        """Matrix multiplication"""


LinearOperatorLike: TypeAlias = LinearOperator | Matrix

__all__ = ["LinearOperatorLike"]
