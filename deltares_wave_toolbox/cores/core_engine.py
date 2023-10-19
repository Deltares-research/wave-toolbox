import numbers
import sys
from typing import Any, Union

import numpy as np
import numpy.typing as npt


def convert_to_array_type(x: npt.ArrayLike) -> npt.NDArray[Any]:
    if isinstance(x, numbers.Number):
        x = np.asarray([x])
    elif not isinstance(x, (np.ndarray)):
        x = np.asarray(x)
    return x


def convert_to_vector(x: npt.ArrayLike) -> tuple[npt.NDArray[Any], tuple[int, int]]:
    x = convert_to_array_type(x)

    # ensure vector convention if 1d array is used either (n,1) or (n,)
    xSize = _size(x)
    if xSize[1] == 1:
        x = x.flatten()

    return x, xSize


def get_parameter_type(x: npt.ArrayLike) -> Union[type[complex], type[float]]:
    if isinstance(x, complex):
        return complex
    else:
        return float


def monotonic_increasing_constant_step(x: npt.ArrayLike) -> bool:
    xDiff = np.diff(x)
    isUniform = bool(np.all((xDiff - xDiff[0]) < 1000000 * sys.float_info.epsilon))
    xMonotonic = bool(np.all(xDiff > 0))

    return xMonotonic and isUniform


def _size(x: npt.ArrayLike) -> tuple[int, int]:
    dimx = 0
    dimy = 0
    if x.size != 0:
        dims = x.shape

        # handle up to two dimensions.
        if len(dims) == 1:
            dimx = x.shape[0]
        elif len(dims) == 2:
            dimx = x.shape[0]
            dimy = x.shape[1]
    return dimx, dimy


def is1darray(x: npt.ArrayLike) -> bool:
    is1d = False
    dimx, dimy = _size(x)
    if dimx != 0 and dimy == 0:
        is1d = True
    return is1d


def approx_array_index(array: npt.ArrayLike, user_value: float) -> int:
    """
    Return index of the array value closest to user specified value.

    Parameters
    ----------
    array : of type ndarray
        array of values.
    user_value : of type float
        value to search for

    Returns
    -------
    idx : of type integer
        index value of array value closest to the user specified value.

    """
    array = np.asarray(array)
    idx = (np.abs(array - user_value)).argmin()
    return int(idx)
