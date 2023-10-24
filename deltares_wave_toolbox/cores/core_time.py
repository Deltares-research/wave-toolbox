import numpy as np
from numpy import float64
from numpy.typing import NDArray

import deltares_wave_toolbox.cores.core_engine as core_engine


def sort_wave_params(
    hWave: NDArray[float64], tWave: NDArray[float64]
) -> tuple[NDArray[float64], NDArray[float64]]:
    """Sorts the wave height and wave period

    This functions sorts the wave heights and wave periods in arrays hWave and tWave, and stores them in arrays
    hWaveSorted and tWaveSorted. The sorting is done such that in hWaveSorted the wave heights of hWave are sorted in
    descending order. This same sorting is applied to tWave, with the result in tWaveSorted. This means that
    hWaveSorted(i) and hWaveSorted(i) correspond to the wave height and wave period of the same wave.

    Parameters
    ----------
    hWave : NDArray[float64]
        1D array containing the wave heights of the individual waves [m]
    tWave : NDArray[float64]
        1D array containing the periods of the individual waves [s]

    Returns
    -------
    tuple[NDArray[float64], NDArray[float64]]
        hWaveSorted : NDArray[float64]
            1D array containing wave heights of the individual waves, sorted in descending order [m]
        tWaveSorted : NDArray[float64]
            1D array containing wave periods of the individual waves, using the same sorting (re-arranging) as
            applied to hWave [s]

    Raises
    ------
    ValueError
        Input error, Input arrays hWave is not 1D
    ValueError
        Input error, Input array tWave is not 1D
    ValueError
        Input arrays hWave and tWave have different length

    Example
    -------
    >>> import numpy as np
    >>> hWave = np.asrange([2,4,3])
    >>> tWave = np.asrange([6,7,8])
    >>> # Then we get:
    >>> [hWaveSorted, tWaveSorted] = sort_wave_params(hWave,tWave)
    >>> #hWaveSorted = np.asrange([4,3,2])
    >>> #tWaveSorted = np.asrange([7,8,6])  (and not: tWaveSorted = [8 7 6])

    """
    # --- ensure input is of type ndarray
    hWave = core_engine.convert_to_array_type(hWave)
    tWave = core_engine.convert_to_array_type(tWave)

    # Check on input arguments
    # --- Check whether the size of the input arguments is identical
    is1d_array = core_engine.is1darray(hWave)
    if not is1d_array:
        raise ValueError("sort_wave_params: Input error, Input arrays hWave is not 1D")

    is1d_array = core_engine.is1darray(tWave)
    if not is1d_array:
        raise ValueError("sort_wave_params: Input error, Input array tWave is not 1D")

    # --- Check whether number of elements in input arguments is identical
    isequal_length = len(hWave) == len(tWave)
    if not isequal_length:
        raise ValueError(
            "InputError", "Input arrays hWave and tWave have different length"
        )

    # --- Number of waves
    nWave = len(hWave)

    # --- If there are no waves, return to calling function
    hWaveSorted = np.empty((0, 0))
    tWaveSorted = np.empty((0, 0))
    if nWave < 1:
        return hWaveSorted, tWaveSorted

    # Computational core
    # --- Sort the wave height in descending order
    iSorted = np.argsort(hWave)[
        ::-1
    ]  # term [::-1] reverses the array, meaning descending order.
    hWaveSorted = hWave[iSorted]
    # --- Sort the wave periods
    tWaveSorted = tWave[iSorted]
    #
    return hWaveSorted, tWaveSorted


def determine_zero_crossing(
    t: NDArray[float64],
    xTime: NDArray[float64],
    typeCross: str = "down",
) -> tuple[int, NDArray[float64]]:
    """Determines zero-crossings (number and positions) of signal.

    This function determines the zero-crossings of a given time signal xTime = xTime(t). Here, t stands for time, and
    xTime for a given signal (for example, time series of measured surface elevation). The type of zero-crossing can
    be either up-crossing or down-crossing.

    Parameters
    ----------
    t : NDArray[float64]
        1D array containing time axis. The numbers in the array t must be increasing and uniformly spaced
        (uniform time step) [s]
    xTime : NDArray[float64]
        1D array containing signal values, i.e. the time series of the signal. The value xxTime(i) must be the signal
        value at time t(i). Usually water surface elevation [m]
    typeCross : str, optional
        Search for up- or down-crossings, by default "down"

    Returns
    -------
    tuple[int, NDArray[float64]]
        nWave : int
            Number of waves in the signal, where one wave corresponds to two successive zero-crossings. Wave i starts
            at time tCross(i), and end at time tCross(i+1) [-]
        tCross : NDArray[float64]
            1D array of length (nWave+1), containing the time of all zero-crossings. The time of the zero-crossings is
            determined by linear interpolation. Note that in case of no zero-crossing, the array tCross is empty. Note
            that in case of one zero-crossing, the number of waves is zero. [s]

    Raises
    ------
    ValueError
        Input error: Input arrays t is not 1D
    ValueError
        Input error: Input arrays x is not 1D
    ValueError
        Input error: Input arrays t and x have different length
    ValueError
        Input error: Wrong input argument for typeCross

    """
    # --- ensure input is of type ndarray
    t = core_engine.convert_to_array_type(t)
    xTime = core_engine.convert_to_array_type(xTime)

    # Perform checks on the input arguments
    is1d_array = core_engine.is1darray(t)
    if not is1d_array:
        raise ValueError("sort_wave_params: Input error: Input arrays t is not 1D")

    is1d_array = core_engine.is1darray(xTime)
    if not is1d_array:
        raise ValueError("sort_wave_params: Input error: Input arrays x is not 1D")

    # --- Check whether the length of arrays t and x the same
    isequal_length = len(t) == len(xTime)
    if not isequal_length:
        raise ValueError(
            "sort_wave_params: Input error: Input arrays t and x have different length"
        )

    # Computational core
    nWave = -1
    tCross = np.empty((0, 0))
    dt = t[1] - t[0]
    nTime = len(t)

    # --- Loop over all time levels, to determine zero-crossings
    if typeCross.lower() == "up":
        # --- Upcrossings
        for iTime in np.arange(0, nTime - 1):
            if xTime[iTime] <= 0 and xTime[iTime + 1] > 0:
                # --- Add 1 to counter number of zero-crossings
                nWave = nWave + 1

                # --- Determine time level of zero-crossing by means of linear
                #     interpolation
                tCrossing = t[iTime] + dt * xTime[iTime] / (
                    xTime[iTime] - xTime[iTime + 1]
                )
                tCross = np.append(tCross, tCrossing)

    elif typeCross.lower() == "down":
        # --- Downcrossings
        for iTime in np.arange(0, nTime - 1):
            if xTime[iTime] >= 0 and xTime[iTime + 1] < 0:
                # --- Add 1 to counter number of zero-crossings
                nWave = nWave + 1

                # --- Determine time level of zero-crossing by means of linear
                #     interpolation
                tCrossing = t[iTime] + dt * xTime[iTime] / (
                    xTime[iTime] - xTime[iTime + 1]
                )
                tCross = np.append(tCross, tCrossing)
    else:
        raise ValueError(
            "sort_wave_params: Input error: Wrong input argument for typeCross "
        )

    # --- If no zero-crossings are found, put number of waves to zero (the
    #     value -1 may be considered inappropriate)
    if nWave == -1:
        nWave = 0

    return nWave, tCross


def highest_waves_params(
    hWaveSorted: NDArray[float64],
    tWaveSorted: NDArray[float64],
    fracP: float,
) -> tuple[float, float]:
    """Computes wave parameters of selection largest waves

    This function computes the wave height hFracP and wave period tFracP by taking the average of the fraction fracP
    of the highest waves. When fracP = 1/3, then hFracP is equal to the significant wave height and tFracP is equal
    to the significant wave period.

    Parameters
    ----------
    hWaveSorted : NDArray[float64]
        1D array containing wave heights of the individual waves, sorted in descending order [m]
    tWaveSorted : NDArray[float64]
        1D array containing wave periods of the individual waves, using the same sorting (re-arranging) as
            applied to hWave [s]
    fracP : float
        fraction. Should be between 0 and 1 [-]

    Returns
    -------
    tuple[float, float]
        hFracP : float
            average of the wave heights of the highest fracP waves [m]
        tFracP : float
            average of the wave periods of the highest fracP waves [s]

    Raises
    ------
    ValueError
        Input error: Input arrays hWaveSorted is not 1D
    ValueError
        Input error: Input arrays tWaveSorted is not 1D
    ValueError
        Input arrays tWaveSorted and hWaveSorted have different length
    ValueError
        Input error: Input parameter fracP should be between 0 and 1
    ValueError
        Input error:Input array hWaveSorted is not correctly sorted

    Example
    -------
    >>> import numpy as np
    >>> hWaveSorted = [7.1, 6.0, 5.8, 4.4, 3.1, 2.2, 1.4]
    >>> tWaveSorted = [5.1, 3.2, 3.7, 0.0, 2.0, 3.0, 1.1]
    >>> [Hsig,Tsig] = highest_waves_params(hWaveSorted,tWaveSorted,1/3)

    """
    # --- ensure input is of type ndarray
    hWaveSorted = core_engine.convert_to_array_type(hWaveSorted)
    tWaveSorted = core_engine.convert_to_array_type(tWaveSorted)

    # Perform checks on the input arguments
    is1d_array = core_engine.is1darray(hWaveSorted)
    if not is1d_array:
        raise ValueError(
            "highest_waves_params: Input error: Input arrays hWaveSorted is not 1D"
        )

    is1d_array = core_engine.is1darray(tWaveSorted)
    if not is1d_array:
        raise ValueError(
            "highest_waves_params: Input error: Input arrays tWaveSorted is not 1D"
        )

    # --- Check whether the length of arrays t and x the same
    isequal_length = len(tWaveSorted) == len(hWaveSorted)
    if not isequal_length:
        raise ValueError(
            "highest_waves_params: Input error: Input arrays tWaveSorted and hWaveSorted have different length"
        )

    # --- Input parameter fracP should be between 0 and 1
    if fracP < 0 or fracP > 1:
        raise ValueError(
            "highest_waves_params: Input error: Input parameter fracP should be between 0 and 1 "
        )

    # --- If there are no waves, return to calling function
    hFracP = 0.0
    tFracP = 0.0

    # --- Number of waves
    nWave = len(hWaveSorted)
    if nWave < 1:
        return hFracP, tFracP

    # --- Check whether hWaveSorted is indeed sorted
    issorted = (hWaveSorted == np.sort(hWaveSorted)[::-1]).all()
    if not issorted:
        raise ValueError(
            "highest_waves_params: Input error:Input array hWaveSorted is not correctly sorted "
        )

    # Computational core
    # --- Number of waves in the fraction that is to be considered
    nWaveP = int(np.floor(nWave * fracP))
    if nWaveP > 0:
        hFracP = float(np.mean(hWaveSorted[0:nWaveP]))
        tFracP = float(np.mean(tWaveSorted[0:nWaveP]))

    return hFracP, tFracP


def exceedance_wave_height(hWaveSorted: NDArray[float64], excPerc: float) -> float:
    """Computes wave height with given exceedance probability

    This function computes the wave height hExcPerc with given exceedance probability percentage excPerc.

    Parameters
    ----------
    hWaveSorted : NDArray[float64]
        1D array containing wave heights of the individual waves, sorted in descending order [m]
    excPerc : float
        exceedance probability percentage. excPerc = 2 means an exceedance percentage of 2%. The value of excPerc
        should not exceed 100, or be smaller than 0 [%]

    Returns
    -------
    float
        hExcPerc : float
            wave height with given exceedance probability [m]

    Raises
    ------
    ValueError
        Input error: Input arrays hWaveSorted is not 1D
    ValueError
        Input error: Input parameter excPerc should be between 0 and 100
    ValueError
        Input error:Input array hWaveSorted is not correctly sorted

    Example
    -------
    >>> import numpy as np
    >>> hWaveSorted = [7.1, 6.0, 5.8, 4.4, 3.1, 2.2, 1.4]
    >>> [hExcPerc_33perc] = exceedance_wave_height(hWaveSorted,33)

    """
    hWaveSorted = core_engine.convert_to_array_type(hWaveSorted)

    # Perform checks on the input arguments
    is1d_array = core_engine.is1darray(hWaveSorted)
    if not is1d_array:
        raise ValueError(
            "exceedance_wave_height: Input error: Input arrays hWaveSorted is not 1D"
        )

    # Check on input arguments
    # --- Input parameter excPerc should not exceed 100 (100%) or be smaller than 0
    isvalid_range = excPerc > 100 or excPerc < 0
    if isvalid_range:
        raise ValueError(
            "exceedance_wave_height: Input error: Input parameter excPerc should be between 0 and 100 "
        )

    # --- If there are no waves, return to calling function
    hExcPerc = 0.0
    # --- Number of waves
    nWave = len(hWaveSorted)
    if nWave < 1:
        return hExcPerc

    # --- Check whether hWaveSorted is indeed sorted
    issorted = (hWaveSorted == np.sort(hWaveSorted)[::-1]).all()
    if not issorted:
        raise ValueError(
            "exceedance_wave_height: Input error:Input array hWaveSorted is not correctly sorted "
        )

    # Computational core
    # --- Index of the wave corresponding to the given exceedance probability percentage
    iWaveP = int(np.floor(nWave * excPerc / 100.0)) - 1

    # --- Compute wave height with exceedance probability
    if not (iWaveP < 0):
        hExcPerc = float(hWaveSorted[iWaveP])

    return hExcPerc


def determine_params_individual_waves(
    tCross: NDArray[float64],
    t: NDArray[float64],
    xTime: NDArray[float64],
) -> tuple[
    NDArray[float64],
    NDArray[float64],
    NDArray[float64],
    NDArray[float64],
    NDArray[float64],
    NDArray[float64],
]:
    """Determines parameters per individual wave

    This function determines several wave properties (wave period, wave height, crest amplitude, trough amplitude, and
    the time at which the maximum crest and trough amplitudes occur) of all individual waves in a wave train. The wave
    train is given by time series xTime = xTime(t), and the zero-crossings occur at time levels given in tCross

    Parameters
    ----------
    tCross : NDArray[float64]
        1D array of length (nWave+1), containing the time of all zero-crossings. The time of the zero-crossings is
        determined by linear interpolation. Note that in case of no zero-crossing, the array tCross is empty. Note
        that in case of one zero-crossing, the number of waves is zero. [s]
    t : NDArray[float64]
        1D array containing time axis. The numbers in the array t must be increasing and uniformly spaced
        (uniform time step) [s]
    xTime : NDArray[float64]
        1D array containing signal values, i.e. the time series of the signal. The value xTime(i) must be the signal
        value at time t(i). Usually water surface elevation [m]

    Returns
    -------
    tuple[ NDArray[float64], NDArray[float64], NDArray[float64], NDArray[float64], NDArray[float64], NDArray[float64], ]
        tWave : NDArray[float64]
            1D array containing the periods of the individual waves [s]
        hWave : NDArray[float64]
            1D array containing the wave heights of the individual waves [m]
        aCrest : NDArray[float64]
            1D array containing the maximum amplitude of the crests of the individual waves [m]
        aTrough : NDArray[float64]
            1D array containing the maximum amplitude of the troughs of the individual waves [m]
        tCrest : NDArray[float64]
            1D array containing the time at which maximum crest amplitude of the individual waves occurs [s]
        tTrough : NDArray[float64]
            1D array containing the time at which maximum trough amplitude of the individual waves occurs [s]

    Notes
    -----
    * All these arrays have a length equal to nWave, which is the number of
    waves in the wave train
    * The values of aTrough are always smaller than zero
    * hWave = aCrest - aTrough

    Raises
    ------
    ValueError
        Input error: Input arrays tCross is not 1D
    ValueError
        Input error: Input arrays t is not 1D
    ValueError
        Input error: Input arrays x is not 1D
    ValueError
        Length of input arrays t and x not identical

    """
    # --- ensure input is of type ndarray
    tCross = core_engine.convert_to_array_type(tCross)
    t = core_engine.convert_to_array_type(t)
    xTime = core_engine.convert_to_array_type(xTime)

    # Perform checks on input arguments
    # --- In the possible situation of no zero-crossings, the array tCross is
    #     empty. In this situation, make output variables empty arrays and
    #     return
    tWave = np.empty((0, 0))
    hWave = np.empty((0, 0))
    aCrest = np.empty((0, 0))
    aTrough = np.empty((0, 0))
    tCrest = np.empty((0, 0))
    tTrough = np.empty((0, 0))
    if tCross.size == 0:
        return hWave, tWave, aCrest, aTrough, tCrest, tTrough

    # Perform checks on the input arguments
    is1d_array = core_engine.is1darray(tCross)
    if not is1d_array:
        raise ValueError(
            "determine_params_individual_waves: Input error: Input arrays tCross is not 1D"
        )

    is1d_array = core_engine.is1darray(t)
    if not is1d_array:
        raise ValueError(
            "determine_params_individual_waves: Input error: Input arrays t is not 1D"
        )

    is1d_array = core_engine.is1darray(xTime)
    if not is1d_array:
        raise ValueError(
            "determine_params_individual_waves: Input error: Input arrays x is not 1D"
        )

    # --- Check whether the length of arrays t and x the same
    isequal_length = len(t) == len(xTime)
    if not isequal_length:
        raise ValueError(
            "determine_params_individual_waves: Length of input arrays t and x not identical "
        )

    # Computational core
    # --- Determine number of waves
    nWave = len(tCross) - 1

    # --- Initialize output arrays (could have used append also, without initializing array)
    tWave = np.zeros((1, nWave), dtype=float)[0]
    hWave = np.zeros((1, nWave), dtype=float)[0]
    aCrest = np.zeros((1, nWave), dtype=float)[0]
    aTrough = np.zeros((1, nWave), dtype=float)[0]
    tCrest = np.zeros((1, nWave), dtype=float)[0]
    tTrough = np.zeros((1, nWave), dtype=float)[0]

    # --- Do a loop over all waves
    for iWave in np.arange(0, nWave):
        # --- Determine initial and end time of wave, and corresponding
        #     positions in time-array
        tIni = tCross[iWave]  # Initial time
        tEnd = tCross[iWave + 1]  # End time
        iIni = np.where(t >= tIni)[0][0]
        iEnd = np.where(t <= tEnd)[0][-1]

        # --- Find position where maximum and minimum occurs, for the
        #     individual wave
        #     Argument 'first' is included, to account for the possible
        #     situation of two extrema with the same function
        #     value in the given wave
        tIwave = t[iIni : iEnd + 1]
        xIwave = xTime[iIni : iEnd + 1]
        iMax = np.where(xIwave == max(xIwave))[0][0]
        iMin = np.where(xIwave == min(xIwave))[0][0]

        # --- Fill the output arrays
        tWave[iWave] = tEnd - tIni
        aCrest[iWave] = xIwave[iMax]
        aTrough[iWave] = xIwave[iMin]
        tCrest[iWave] = tIwave[iMax]
        tTrough[iWave] = tIwave[iMin]
        hWave[iWave] = aCrest[iWave] - aTrough[iWave]

    return hWave, tWave, aCrest, aTrough, tCrest, tTrough
