import numpy as np
import numbers
import sys


def convert_to_array_type(x):
    if isinstance(x, numbers.Number):
        x = np.asarray([x])
    elif not isinstance(x, (np.ndarray)):
        x = np.asarray(x)
    return x


def convert_to_vector(x):
    if isinstance(x, numbers.Number):
        x = np.asarray([x])
    elif not isinstance(x, (np.ndarray)):
        x = np.asarray(x)

    # ensure vector convention if 1d array is used either (n,1) or (n,)
    xSize = _size(x)
    if xSize[1] == 1:
        x = x.flatten()

    return x, xSize


def get_parameter_type(x):
    if isinstance(x, complex):
        return complex
    else:
        return float


def monotonic_increasing_constant_step(x):
    xDiff = np.diff(x)
    isUniform = np.all((xDiff - xDiff[0]) < 1000000 * sys.float_info.epsilon)  #
    xMonotonic = np.all(xDiff > 0)

    return xMonotonic and isUniform


def _size(x):
    dimx = 0
    dimy = 0
    if x is not None:
        dims = x.shape

        # handle up to two dimensions.
        if len(dims) == 1:
            dimx = x.shape[0]
        elif len(dims) == 2:
            dimx = x.shape[0]
            dimy = x.shape[1]
    return dimx, dimy


def isempty(x):
    is_empty = False
    dimx, dimy = _size(x)
    if dimx == 0 and dimy == 0:
        is_empty = True
    return is_empty


def is1darray(x):
    is1d = False
    dimx, dimy = _size(x)
    if dimx != 0 and dimy == 0:
        is1d = True
    return is1d


def approx_array_index(array, user_value: float):
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
    return idx
