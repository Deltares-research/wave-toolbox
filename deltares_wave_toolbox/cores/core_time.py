import numpy as np
import deltares_wave_toolbox.cores.core_engine as core_engine


def sort_wave_params(hWave=None, tWave=None):
    """
    SORT_WAVE_PARAMS  Sorts the wave height and wave period
    (WAVELAB: sortwaveparams)

    This functions sorts the wave heights and wave periods in arrays
    hWave and tWave, and stores them in arrays hWaveSorted and tWaveSorted.
    The sorting is done such that in hWaveSorted the wave heights of hWave
    are sorted in descending order. This same sorting is applied to
    tWave, with the result in tWaveSorted. This means that hWaveSorted(i) and
    hWaveSorted(i) correspond to the wave height and wave period of the
    same wave.

    Subject: time domain analysis of waves


    Parameters
    ----------
    hWave    : array double (1D)
             1D array containing the wave heights of the individual waves
    tWave    : array double (1D)
             1D array containing the periods of the individual waves

    Returns
    -------
    hWaveSorted : array double (1D)
                1D array containing wave heights, sorted in descending
                order
    tWaveSorted : array double (1D)
                1D array containing wave periods, using the same sorting
                (re-arranging) as applied to hWave
    Note:
         * all input and output arrays have the same size
    Syntax:
          [hWaveSorted,tWaveSorted] = sort_wave_params(hWave,tWave)


    Example:
    >>> import numpy as np
    >>> hWave = np.asrange([2,4,3])
    >>> tWave = np.asrange([6,7,8])
    >>> # Then we get:
    >>> [hWaveSorted, tWaveSorted] = sort_wave_params(hWave,tWave)
    >>> #hWaveSorted = np.asrange([4,3,2])
    >>> #tWaveSorted = np.asrange([7,8,6])  (and not: tWaveSorted = [8 7 6])

    See also determineparamsindividualwaves, highestwavesparams

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
            "WaveLab:InputError", "Input arrays hWave and tWave have different length"
        )

    # --- Number of waves
    nWave = len(hWave)

    # --- If there are no waves, return to calling function
    hWaveSorted = []
    tWaveSorted = []
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


def determine_zero_crossing(t=None, x=None, typeCross="down"):
    """
    DETERMINEZEROCROSSING  Determines zero-crossings (number and positions) of signal.
    This function determines the zero-crossings of a given time signal x =
    x(t). Here, t stands for time, and x for a given signal (for example,
    time series of measured surface elevation). The type of zero-crossing
    can be either up-crossing or down-crossing.
    (WAVELAB: sortwaveparams)

    Subject: time domain analysis of waves


    Parameters
    ----------
    t         : array double (1D)
              1D real array containing time values. The numbers in the
              array t must be increasing and uniformly spaced (uniform
              time step)
    x         : array double (1D)
              1D real array containing signal values, i.e. the time
              series of the signal. The value x(i) must be the signal
              value at time t(i)
    typeCross : string
              character indicating which type of zero-crossings is
              requested.
              There are two options:
                * typeCross = 'up', for up-crossings
                * typeCross = 'down', for down-crossings


    Returns
    -------
    nWave     : array double (1D)
              integer indicating the number of waves in the signal, where
              one wave corresponds to two successive zero-crossings. Wave
              i start at time tCross(i), and end at time tCross(i+1).
    tCross    : array double (1D)
              1D array of length (nWave+1), containing the time of all
              zero-crossings. The time of the zero-crossings is
              determined by linear interpolation.
              Note that in case of no zero-crossing, the array tCross is
              empty.
              Note that in case of one zero-crossing, the number of waves
              is zero

    Syntax:
         [nWave,tCross] = determine_zero_crossing(t,x,typeCross)


    Example:
    >>>
    >>>   [numberOfWaves,timeUpCrossings] = determinezerocrossing(time,surfelev,'up')

    See also determineParamsIndividualWaves
    """

    # --- ensure input is of type ndarray
    t = core_engine.convert_to_array_type(t)
    x = core_engine.convert_to_array_type(x)

    # Perform checks on the input arguments
    is1d_array = core_engine.is1darray(t)
    if not is1d_array:
        raise ValueError("sort_wave_params: Input error: Input arrays t is not 1D")

    is1d_array = core_engine.is1darray(x)
    if not is1d_array:
        raise ValueError("sort_wave_params: Input error: Input arrays x is not 1D")

    # --- Check whether the length of arrays t and x the same
    isequal_length = len(t) == len(x)
    if not isequal_length:
        raise ValueError(
            "sort_wave_params: Input error: Input arrays t and x have different length"
        )

    # Computational core
    nWave = -1
    tCross = []
    dt = t[1] - t[0]
    nTime = len(t)

    # --- Loop over all time levels, to determine zero-crossings
    if typeCross.lower() == "up":
        # --- Upcrossings
        for iTime in np.arange(0, nTime - 1):
            if x[iTime] <= 0 and x[iTime + 1] > 0:
                # --- Add 1 to counter number of zero-crossings
                nWave = nWave + 1

                # --- Determine time level of zero-crossing by means of linear
                #     interpolation
                tCrossing = t[iTime] + dt * x[iTime] / (x[iTime] - x[iTime + 1])
                tCross.append(tCrossing)

    elif typeCross.lower() == "down":
        # --- Downcrossings
        for iTime in np.arange(0, nTime - 1):
            if x[iTime] >= 0 and x[iTime + 1] < 0:
                # --- Add 1 to counter number of zero-crossings
                nWave = nWave + 1

                # --- Determine time level of zero-crossing by means of linear
                #     interpolation
                tCrossing = t[iTime] + dt * x[iTime] / (x[iTime] - x[iTime + 1])
                tCross.append(tCrossing)
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
    hWaveSorted=None, tWaveSorted=None, fracP=None
) -> list[float] | list[float]:
    """
    HIGHEST_WAVES_PARAMS  Computes wave parameters of selection largest waves
    (WAVELAB: highestwavesparams)

    This function computes the wave height hFracP and wave period tFracP by
    taking the average of the fraction fracP of the highest waves. When
    fracP = 1/3, then hFracP is equal to the significant wave height and
    tFracP is equal to the significant wave period.

    Subject: time domain analysis of waves



    Parameters
    ----------
    hWaveSorted : array double (1D)
                1D array containing wave heights, sorted in descending
                order
    tWaveSorted : array double (1D)
                1D array containing wave periods, using the same sorting
                (re-arranging) as applied to hWave
    fracP       : double
                fraction. Should be between 0 and 1

    Returns
    -------
    hFracP     : double
               average of the wave heights of the highest fracP waves
    tFracP     : double
               average of the wave periods of the highest fracP waves

    Syntax:
         [hFracP,tFracP] = highest_waves_params(hWaveSorted,tWaveSorted,fracP)

    Example:
    >>> import numpy as np
    >>> hWaveSorted = [7.1, 6.0, 5.8, 4.4, 3.1, 2.2, 1.4]
    >>> tWaveSorted = [5.1, 3.2, 3.7, 0.0, 2.0, 3.0, 1.1]
    >>> [Hsig,Tsig] = highest_waves_params(hWaveSorted,tWaveSorted,1/3)

    See also determine_params_individual_waves, sort_wave_params,
             exceedance_wave_height

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
    hFracP = None
    tFracP = None

    # --- Number of waves
    nWave = len(hWaveSorted)
    if nWave < 1:
        return [hFracP, tFracP]

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
        hFracP = determine_mean(hWaveSorted[0:nWaveP])
        tFracP = determine_mean(tWaveSorted[0:nWaveP])

    return hFracP, tFracP


def exceedance_wave_height(hWaveSorted=None, excPerc=None) -> float:
    """
    EXCEEDANCEWAVEHEIGHT  Computes wave height with given exceedance probability
    (WAVELAB:exceedancewaveheight)

    This function computes the wave height hExcPerc with given exceedance
    probability percentage excPerc.

    Subject: time domain analysis of waves


    Parameters
    ----------
    hWaveSorted : array double (1D)
                1D array containing wave heights, sorted in descending
                order
    excPerc     :
                exceedance probability percentage. excPerc = 2 means an
                exceedance percentage of 2%. The value of excPerc should
                not exceed 100, or be smaller than 0

    Returns
    -------
    hExcPerc  : double
               wave height with given exceedance probability


    Syntax:
          [hExcPerc] = exceedance_wave_height(hWaveSorted,excPerc)


    Example:
    >>> import numpy as np
    >>> hWaveSorted = [7.1, 6.0, 5.8, 4.4, 3.1, 2.2, 1.4]
    >>> [hExcPerc_33perc] = exceedance_wave_height(hWaveSorted,33)

    See also determine_params_individual_waves, sort_wave_params,
    highest_waves_params

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
    hExcPerc = None
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


# ? Why oh why do we need to implement this ourselves???
def determine_mean(qWave=None) -> float | Any:
    """
    DETERMINE_MEAN  Determines mean of quantity
    (WAVELAB: determinemean)

    This function determines the mean of a quantity, for example the mean
    wave height or the mean wave period. With qWave
    an array [q_1,...,q_nWave], with q_i pertaining to wave i and
    nWave the number of waves, the mean is given by:
       qMean = sum q_i / nWave
    where the sum is over all wave heights.

    Subject: time domain analysis of waves

    Parameters
    ----------
    qWave    : array double (1D)
             1D array containing parameters of individual waves, for example
             wave height or wave period

    Returns
    -------
    qMean    : doiuble
             mean of qWave

    Syntax:
          qMean = determine_mean( qWave )


    Example:
    >>> import numpy as np
    >>>  hWave = [1.0, 1.2, 0.8, 1.1]  # Wave heights of individual waves
    >>>  tWave = [2.3, 2.1, 1.8, 2.0]   # Wave periods of individual waves
    >>>  hMean   = determine_mean( hWave )
    >>>  tMean   = determine_mean( tWave )

    See also determine_params_individualwaves, determine_hrms

    """
    # --- ensure input is of type ndarray
    qWave = core_engine.convert_to_array_type(qWave)

    # Perform checks on the input arguments
    is1d_array = core_engine.is1darray(qWave)
    if not is1d_array:
        raise ValueError("determine_mean: Input error: Input arrays qWave is not 1D")

    # Check on input arguments
    # --- Number of waves
    nWave = len(qWave)

    # --- If there are no waves, return to calling function
    qMean = np.nan
    if nWave < 1:
        return qMean

    # Comutational core
    # --- Compute mean
    qMean = sum(qWave) / nWave

    return qMean


def determine_params_individual_waves(tCross=None, t=None, x=None):
    """
    DETERMINE_PARAMS_INDIVIDUALWAVES  Determines parameters per individual wave
    (WAVELAB: determineparamsindividualwaves)

    This function determines several wave properties (wave period, wave
    height, crest amplitude, trough amplitude, and the time at which the
    maximum crest and trough amplitudes occur) of all individual waves in a
    wave train. The wave train is given by time series x = x(t), and the
    zero-crossings occur at time levels given in tCross

    Subject: time domain analysis of waves

    Parameters
    ----------
    tCross    : array double (1D)
              1D array of length (nWave+1), containing the time of all
              zero-crossings. nWave is an integer representing the
              number of waves.
    t         : array double (1D)
              1D real array containing time values. The numbers in the
              array t must be increasing and uniformly spaced (uniform
              time step)
    x         : array double (1D)
              1D real array containing signal values, i.e. the time
              series of the signal. The value x(i) must be the signal
              value at time t(i)

    Returns
    -------
    tWave    : array double (1D)
             1D array containing the periods of the individual waves
    hWave    : array double (1D)
             1D array containing the wave heights of the individual waves
    aCrest   : array double (1D)
             1D array containing the maximum amplitudes of the crest of
             the individual waves
    aTrough  : array double (1D)
             1D array containing the maximum amplitudes of the trough of
             the individual waves
    tCrest   : array double (1D)
             1D array containing the time at which maximum crest
             amplitude of the individual waves occurs
    tTrough  : array double (1D)
             1D array containing the time at which maximum trough
             amplitude of the individual waves occurs
    Notes:
         * All these arrays have a length equal to nWave, which is the number of
           waves in the wave train
         * The values of aTrough are always smaller than zero
         * hWave = aCrest - aTrough
    Syntax:
          [hWave,tWave,aCrest,aTrough,tCrest,tTrough] =
                  determine_params_individual_waves(tCross,t,x)

    Example:
    >>> import numpy as np
    >>>

    See also determinezerocrossing

    """

    # --- ensure input is of type ndarray
    tCross = core_engine.convert_to_array_type(tCross)
    t = core_engine.convert_to_array_type(t)
    x = core_engine.convert_to_array_type(x)

    # Perform checks on input arguments
    # --- In the possible situation of no zero-crossings, the array tCross is
    #     empty. In this situation, make output variables empty arrays and
    #     return
    tWave = []
    hWave = []
    aCrest = []
    aTrough = []
    tCrest = []
    tTrough = []
    if core_engine.isempty(tCross):
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

    is1d_array = core_engine.is1darray(x)
    if not is1d_array:
        raise ValueError(
            "determine_params_individual_waves: Input error: Input arrays x is not 1D"
        )

    # --- Check whether the length of arrays t and x the same
    isequal_length = len(t) == len(x)
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
        xIwave = x[iIni : iEnd + 1]
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


def test_doctstrings() -> None:
    import doctest

    doctest.testmod()


if __name__ == "__main__":
    test_doctstrings()
