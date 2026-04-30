import numpy as np
import numpy.typing as npt


def log0(x: npt.NDArray) -> npt.NDArray:
    """Compute the element-wise logarithm of an array, treating log(0) as 0.

    Parameters
    ----------
    x
        Input array.

    Returns
    -------
    npt.NDArray
        Element-wise logarithm of the input array, with log(0) treated as 0.

    Example
    -------
    >>> import numpy as np
    >>> from raking_codcorrect.special import log0
    >>> x = np.array([1.0, 0.0])
    >>> log0(x)
    array([0., 0.])

    """
    index = x != 0.0
    out = np.zeros_like(x)
    out[index] = np.log(x[index])
    return out


def div0(numerator: npt.NDArray, denominator: npt.NDArray) -> npt.NDArray:
    """Compute the element-wise division of two arrays, treating 0/any as 0.

    Parameters
    ----------
    numerator
        Numerator array.
    denominator
        Denominator array.

    Returns
    -------
    npt.NDArray
        Element-wise division of the numerator by the denominator, with 0/any
        treated as 0.

    Example
    -------
    >>> import numpy as np
    >>> from raking_codcorrect.special import div0
    >>> numerator = np.array([1.0, 0.0, 0.0])
    >>> denominator = np.array([2.0, 0.0, np.nan])
    >>> div0(numerator, denominator)
    array([0.5, 0. , 0.])

    """
    index = numerator != 0
    out = np.zeros_like(numerator)
    if not np.isfinite(denominator[index]).all():
        raise ValueError(
            "Denominator has non-finite values where numerator is non-zero"
        )
    if not (denominator[index] != 0).all():
        raise ValueError("Denominator has zero values where numerator is non-zero")
    out[index] = numerator[index] / denominator[index]
    return out


def mul0(left: npt.NDArray, right: npt.NDArray) -> npt.NDArray:
    """Compute the element-wise multiplication of two arrays, treating 0*any or
    any*0 as 0.

    Parameters
    ----------
    left
        Left array.
    right
        Right array.

    Returns
    -------
    npt.NDArray
        Element-wise multiplication of the two input arrays, with 0*any or any*0
        treated as 0.

    Example
    -------
    >>> import numpy as np
    >>> from raking_codcorrect.special import mul0
    >>> left = np.array([1.0, np.nan, 0.0])
    >>> right = np.array([2.0, 0.0, np.nan])
    >>> mul0(left, right)
    array([2., 0., 0.])

    """
    index = (left != 0) & (right != 0)
    out = np.zeros_like(left)
    out[index] = left[index] * right[index]
    return out
