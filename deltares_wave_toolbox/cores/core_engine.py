# SPDX-License-Identifier: GPL-3.0-or-later
import numbers
import sys
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def convert_to_array_type(x: ArrayLike) -> NDArray[Any]:
    """Converts input to numpy array

    Parameters
    ----------
    x : ArrayLike
        input, convertible to array

    Returns
    -------
    NDArray[Any]
        numpy array
    """
    if isinstance(x, numbers.Number):
        x = np.asarray([x])
    elif not isinstance(x, np.ndarray):
        x = np.asarray(x)
    return x


def convert_to_vector(x: ArrayLike) -> tuple[NDArray[Any], tuple[int, int]]:
    """Convert input to 1D numpy array

    Parameters
    ----------
    x : ArrayLike
        input, convertible to array

    Returns
    -------
    tuple[NDArray[Any], tuple[int, int]]
        x: NDArray[Any]
            numpy array
        xSize: tuple[int, int]
            size of the numpy array
    """
    x = convert_to_array_type(x)

    # ensure vector convention if 1d array is used either (n,1) or (n,)
    xSize = _size(x)
    if xSize[1] == 1:
        x = x.flatten()

    return x, xSize


def monotonic_increasing_constant_step(x: ArrayLike) -> bool:
    """Check whether vector is monotonic increasing with constant step size

    Parameters
    ----------
    x : ArrayLike
        input vector

    Returns
    -------
    bool
        True when vector is monotonic increasing with constant step size
    """
    xDiff = np.diff(x)
    isUniform = bool(np.all((xDiff - xDiff[0]) < 1000000 * sys.float_info.epsilon))
    xMonotonic = bool(np.all(xDiff > 0))

    return xMonotonic and isUniform


def _size(x: ArrayLike) -> tuple[int, int]:
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


def is1darray(x: ArrayLike) -> bool:
    """Check is input is a 1D array

    Parameters
    ----------
    x : ArrayLike
        input, convertible to array

    Returns
    -------
    bool
        True if input is a 1D array
    """
    is1d = False
    dimx, dimy = _size(x)
    if dimx != 0 and dimy == 0:
        is1d = True
    return is1d


def approx_array_index(array: ArrayLike, user_value: float) -> int:
    """Return index of the array value closest to user specified value.

    Parameters
    ----------
    array : ArrayLike
        array of values
    user_value : float
        value to search for

    Returns
    -------
    int
        index value of array value closest to the user specified value
    """
    array = np.asarray(array)
    idx = (np.abs(array - user_value)).argmin()
    return int(idx)
