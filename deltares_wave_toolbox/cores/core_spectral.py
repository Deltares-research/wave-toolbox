# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import copy
import math

import numpy as np
from numpy import complex128, float64
from numpy.typing import NDArray
from scipy.signal import welch

import deltares_wave_toolbox.cores.core_engine as core_engine
import deltares_wave_toolbox.series as series


def frequency_averaging(
    f: NDArray[float64],
    sVarDens: NDArray[float64],
    dfDesired: float = 0.0,
) -> tuple[NDArray[float64], NDArray[float64]]:
    """Band averaging of given variance density spectrum

    This function performs a band averaging on a given variance density
    spectrum sVarDens = sVarDens(f) on a frequency axis f onto a coarser
    frequency axis with frequency spacing dfDesired.

    Parameters
    ----------
    f : NDArray[float64]
        1D array containing frequency values, for folded Fourier transform. The number of elements in array f is
        close to half the number of elements in array xTime. To be precise, length(f) = floor(nT/2) + 1, with nT the
        number of elements in array xTime [Hz]
    sVarDens : NDArray[float64]
        1D array containing variance density spectrum of the signal [m^2/Hz]
    dfDesired : float, optional
        desired frequency spacing in Hertz on which the wave spectrum must be computed. If this parameter is omitted,
        then dfDesired = f(1) - f(0), by default 0.0 [Hz]

    Returns
    -------
    tuple[NDArray[float64], NDArray[float64]]
        f_coarse : NDArray[float64]
            frequency axis of computed spectrum. The frequency spacing is (close to) dfDesired [Hz]
        sVarDens_coarse : NDArray[float64]
            band averaged variance density spectrum of the signal on frequency axis f_coarse [m^2/Hz]

    Raises
    ------
    ValueError
        frequency_averaging: Input error: input should be 1d arrays

    Example
    -------
    >>> import numpy as np
    >>> df        = 0.01 #Hz
    >>> f         = np.arange(0,100)*df
    >>> sVarDens  = np.random.normal(loc=1, scale=2, size=len(f))
    >>> dfDesired = 0.02
    >>> f_coarse,sVarDens_coarse = frequency_averaging(f,sVarDens,dfDesired)

    """
    # convert input to array type to be able to handle input like e.g. f = [0.2,0.4]
    f, fSize = core_engine.convert_to_vector(f)
    sVarDens, sFreqSize = core_engine.convert_to_vector(sVarDens)

    if fSize[1] > 1 or sFreqSize[1] > 1:
        raise ValueError("frequency_averaging: Input error: input should be 1d arrays")

    # --- Determine some coefficients
    nFine = fSize[0]
    nFactor = 1
    if dfDesired != 0.0:
        nFactor = np.round(dfDesired / (f[1] - f[0]))

    # --- Avoid nFactor being equal to zero, which may occur if
    #     dfDesired < 0.5 * (f2 - f1)
    nFactor = max(nFactor, 1)
    nCoarse = math.floor(nFine / nFactor)

    # --- Initialize arrays
    f_coarse = np.zeros(len(f[0:nCoarse]))
    sVarDens_coarse = np.zeros(len(f_coarse), dtype=type(sVarDens[0]))

    # --- Perform the averaging
    for iFreq in np.arange(0, nCoarse):  # before np.arange(0,nCoarse)
        ilow = int((iFreq) * nFactor)
        ihigh = int((iFreq + 1) * nFactor)

        f_coarse[iFreq] = np.mean(f[ilow:ihigh])
        sVarDens_coarse[iFreq] = np.mean(sVarDens[ilow:ihigh])

    return f_coarse, sVarDens_coarse


def unfold_spectrum(
    f: NDArray[float64], xFreq: NDArray[complex128], isOdd: bool
) -> tuple[NDArray[float64], NDArray[complex128]]:
    """Unfolds a folded discrete Fourier transform

    This function unfolds a folded discrete Fourier transform xFreq that is given at frequency axis f. Note that this
    frequency axis goes up to the Nyquist frequency. Parameter isOdd indicates whether the underlying original time
    signal - of which xFreq is the discrete Fourier transform - has even (isOdd=0) or odd (isOdd=1) time points.

    The unfolded discrete Fourier xFreqTot, at frequency axis fTot, can be inverted back to time domain using
    function freq2time

    Parameters
    ----------
    f : NDArray[float64]
        1D array containing frequency values, for folded Fourier transform. The number of elements in array f is
        close to half the number of elements in array xTime. To be precise, length(f) = floor(nT/2) + 1, with nT the
        number of elements in array xTime [Hz]
    xFreq : NDArray[complex128]
        1D array (complex!) containing the folded Fourier coefficients of xTime. The value xFreq(i) must be the
        Fourier coefficient at frequency f(i). The number of elements in f and xFreq are the same.
    isOdd : bool
        logical indicating whether nT, the number of time points in original time series, is even (isOdd=False) or
        odd (isOdd=True)

    Returns
    -------
    tuple[NDArray[float64], NDArray[complex128]]
        fTot : NDArray[float64]
            1D array containing frequency values, for unfolded Fourier transform. The frequency axis runs from 0 to
            twice the Nyquist frequency. Array fTot contains as many elements as the original time series. [Hz]
        xFreqTot : NDArray[complex128]
            1D array (complex!) containing the unfolded Fourier coefficients of original time series. The value
            xFreqTot(i) must be the Fourier coefficient at frequency fTot(i). The number of elements in fTot and
            xFreqTot are the same.

    Raises
    ------
    ValueError
        Input error: input should be 1d arrays

    Example
    -------
    >>> import numpy as np
    >>> dt            = 0.1 # s
    >>> t             = np.arange(0,100) *dt  # Time axis
    >>> a1 = 0.5; w1  = 2*np.pi/5; phi1 = 0.35  # Wave component 1
    >>> a2 = 0.7; w2  = 2*np.pi/6; phi2 = 0.96  # Wave component 2
    >>> y = a1*np.cos(w1*t+phi1) + a2*np.cos(w2*t+phi2)
    >>> # --- Compute folded discrete Fourier transform
    >>> f,yFreq,isOdd = time2freq_nyquist(t,y);
    >>> # --- Unfold the discrete Fourier transform
    >>> fTot,yFreqTot = unfold_spectrum(f,yFreq,isOdd)
    >>> # --- Return to time domain
    >>> yTime         = freq2time(yFreqTot)  # yTime must be identical to y

    """
    # convert input to array type to be able to handle input like e.g. f = [0.2,0.4]
    f, fSize = core_engine.convert_to_vector(f)
    xFreq, xFreqSize = core_engine.convert_to_vector(xFreq)

    if fSize[1] > 1 or xFreqSize[1] > 1:
        raise ValueError("unfold_spectrum: Input error: input should be 1d arrays")

    nF = fSize[0]
    nFTot = (nF - 1) * 2 + isOdd

    # --- Construct frequency axis
    # Note that the first half part of the frequency axis is equal to the input
    # frequency axis as given in freq
    df = f[1] - f[0]
    fTot = np.arange(0, nFTot) * df

    xFreqTot = np.zeros((len(fTot)), dtype=complex)
    xFreqTot[0:nF] = copy.deepcopy(xFreq[0:nF])

    # Arrays f and xFreqTot are column vectors. Apply a flip upside-down
    xFreqTot[(nF - 1) + isOdd : nFTot] = np.flipud(np.conj(xFreqTot[1:nF]))

    return fTot, xFreqTot


def coherence(
    f: NDArray[float64],
    xFreq1: NDArray[complex128],
    xFreq2: NDArray[complex128],
    dfDesired: float = 0.0,
) -> tuple[NDArray[float64], NDArray[float64]]:
    """Function to compute the coherence in spectral domain.

    This function computes the coherence (magnitude-squared cohorence) of two complex spectral signal
    xFreq1 = xFreq1(f) and xFreq2 = xFreq2(f), given on frequency axis f. The output coh2 = coh2(f_coh2) is given on
    a frequency axis with frequency resolution dfDesired. Note: the coherence is real, and always between 0 and 1.

    Parameters
    ----------
    f : NDArray[float64]
        1D array containing frequency values, for folded Fourier transform. The number of elements in array f is
        close to half the number of elements in array xTime. To be precise, length(f) = floor(nT/2) + 1, with nT the
        number of elements in array xTime [Hz]
    xFreq1 : NDArray[complex128]
        1D array (complex!) containing the folded Fourier coefficients of signal 1. The value xFreq(i) must be the
        Fourier coefficient at frequency f(i). The number of elements in f and xFreq1 are the same.
    xFreq2 : NDArray[complex128]
        1D array (complex!) containing the folded Fourier coefficients of signal 2. The value xFreq(i) must be the
        Fourier coefficient at frequency f(i). The number of elements in f and xFreq2 are the same.
    dfDesired : float, optional
        desired frequency spacing in Hertz on which the wave spectrum must be computed. If this parameter is omitted,
        then dfDesired = f(1) - f(0), by default 0.0 [Hz]

    Returns
    -------
    tuple[NDArray[float64], NDArray[float64]]
        f_coh2 : NDArray[float64]
            frequency axis of coherence. The frequency spacing is (close to) dfDesired [Hz]
        coh2 : NDArray[float64]
            coherence (magnitude-squared coherence)
    """
    # --- Compute auto-spectral and cross-spectral (absolute value) densities
    #     on original frequency axis
    #     Note: Normalisation factors (2 / (df * Ntime * Ntime)) may be omitted,
    #     since they cancel out
    S11 = abs(xFreq1 * np.conj(xFreq1))
    S22 = abs(xFreq2 * np.conj(xFreq2))
    S12 = abs(xFreq1 * np.conj(xFreq2))

    # --- Apply band averaging
    f_coh2, S11_b = frequency_averaging(f, S11, dfDesired)
    f_coh2, S22_b = frequency_averaging(f, S22, dfDesired)
    f_coh2, S12_b = frequency_averaging(f, S12, dfDesired)

    # --- Omit division by zero, by putting the zeros (very small numbers) to a small
    #     number.
    small_number = 1e-10

    ismall = np.where(S11_b < small_number)[0]
    S11_b[ismall] = small_number

    ismall = np.where(S22_b < small_number)[0]
    S22_b[ismall] = small_number

    # --- Compute coherence
    coh2 = (S12_b) ** 2 / (S11_b * S22_b)

    return f_coh2, coh2


def freq2time(xFreq: NDArray[complex128]) -> NDArray[float64]:
    """Transforms (unfolded) discrete Fourier transform back to time signal

    This function transforms a given discrete and unfolded Fourier transform xFreq (in general a complex quantity!)
    back to time domain. Note that the input Fourier transform xFreq must be unfolded. A given folded Fourier
    transform can be unfolded using the function unfoldspectrum.

    Parameters
    ----------
    xFreq : NDArray[complex128]
        1D array (complex!) containing the folded Fourier coefficients of xTime. The value xFreq(i) must be the
        Fourier coefficient at frequency f(i). The number of elements in f and xFreq are the same.

    Returns
    -------
    NDArray[float64]
        1D array containing signal values, i.e. the time series of the signal. The value xTime(i) must be the signal
        value at time t(i). Usually water surface elevation [m]

    Raises
    ------
    ValueError
        Input error: input should be 1d arrays

    Example
    -------
    >>> import numpy as np
    >>> # --- Create time signal
    >>> dt =0.1   # s
    >>> t = np.arange(0,100)*dt  # Time axis
    >>> a1 = 0.5; w1 = 2*np.pi/5; phi1 = 0.35  # Wave component 1
    >>> a2 = 0.7; w2 = 2*np.pi/6; phi2 = 0.96  # Wave component 2
    >>> y = a1*np.cos(w1*t+phi1) + a2*np.cos(w2*t+phi2)
    >>> # --- Compute discrete Fourier transform
    >>> f,yFreq = time2freq(t,y)
    >>> # --- Return to time domain
    >>> yTime = freq2time(yFreq)  # yTime must be identical to y

    """
    xFreq, xFreqSize = core_engine.convert_to_vector(xFreq)

    # Check on input arguments
    #  xFreq must be a 1D array
    if xFreqSize[1] > 1:
        raise ValueError("freq2time: Input error: input should be 1d arrays")

    # Computational core
    nF = xFreqSize[0]
    xTime = np.real(np.fft.ifft(xFreq, nF))

    return xTime


def time2freq(
    t: NDArray[float64], xTime: NDArray[float64]
) -> tuple[NDArray[float64], NDArray[complex128]]:
    """Computes the discrete Fourier transform coefficients (on unfolded frequency axis) of a given of time signal

    This function computes the Fourier coefficients xFreq (in general complex quantities!) on frequency axis f
    (hence, xFreq = xFreq(f)), from a given time signal Xtime on time axis t (hence, xTime = xTime(t)).

    The Fourier transform is not folded. This means, that the number of elements in arrays f and xFreq is identical
    to nT (nT: the number of elements in arrays t and xTime). Note that the Fourier coefficients in xFreq have a
    complex conjugate symmetry around the Nyquist frequency. Transforming the signal xFreq = xFreq(f) back to time
    domain can be done with the function freq2time

    Parameters
    ----------
    t : NDArray[float64]
        1D real array containing time values. The numbers in the array t must be increasing and uniformly spaced
        (uniform time step). The initial time t(1) can be any value (so it is not obligatory to have t(1) = 0)
    xTime : NDArray[float64]
        1D array containing signal values, i.e. the time series of the signal. The value xTime(i) must be the signal
        value at time t(i). Usually water surface elevation [m]

    Returns
    -------
    tuple[NDArray[float64], NDArray[complex128]]
        f : NDArray[float64]
            1D array containing frequency values, for folded Fourier transform. The number of elements in array f is
            close to half the number of elements in array xTime. To be precise, length(f) = floor(nT/2) + 1, with nT
            the number of elements in array xTime [Hz]
        xFreq : NDArray[complex128]
            1D array (complex!) containing the unfolded Fourier coefficients of xTime. The value xFreq(i) must be the
            Fourier coefficient at frequency f(i). The number of elements in f and xFreq are the same. This number is
            the same as the number of elements in t and xTime.

    Raises
    ------
    ValueError
        Input error: input should be 1d arrays
    ValueError
        Input error: time input parameter must be monotonic with constant step size
    ValueError
        Input error: array sizes differ in dimension

    Example
    -------
    >>> import numpy as np
    >>> # --- Create time signal
    >>> dt =0.1 # s
    >>> t = np.arange(0,100)*dt  # Time axis
    >>> a1 = 0.5; w1 = 2*np.pi/5; phi1 = 0.35  # Wave component 1
    >>> a2 = 0.7; w2 = 2*np.pi/6; phi2 = 0.96  # Wave component 2
    >>> y = a1*np.cos(w1*t+phi1) + a2*np.cos(w2*t+phi2)
    >>> # --- Compute discrete Fourier transform
    >>> f,yFreq = time2freq(t,y)

    """
    t, tSize = core_engine.convert_to_vector(t)
    xTime, xTimeSize = core_engine.convert_to_vector(xTime)

    if tSize[1] > 1 or xTimeSize[1] > 1:
        raise ValueError("time2freq: Input error: input should be 1d arrays")

    if not core_engine.monotonic_increasing_constant_step(t):
        raise ValueError(
            "time2freq: Input error: time input parameter must be monotonic with constant step size"
        )

    if not (tSize[0] == xTimeSize[0]):
        raise ValueError("time2freq: Input error: array sizes differ in dimension")

    # Computational core
    # --- Initialize some constants
    #     Note: T = nT*dt = t(end) - t(1) + dt
    nT = tSize[0]
    dt = t[1] - t[0]
    T = nT * dt

    # --- Create frequency axis
    df = 1.0 / T
    f = np.arange(0, nT) * df  # exlcude f=0

    # --- Compute Fouriertransform
    xFreq = np.fft.fft(xTime)

    return f, xFreq


def time2freq_nyquist(
    t: NDArray[float64], xTime: NDArray[float64]
) -> tuple[NDArray[float64], NDArray[complex128], bool]:
    """Computes the discrete Fourier transform coefficients (on folded frequency axis) of a given of time signal

    This function computes the Fourier coefficients xFreq (in general complex quantities!) on frequency axis f
    (hence, xFreq = xFreq(f)), from a given time signal Xtime on time axis t (hence, xTime = xTime(t)).

    The Fourier transform is folded. This means, that the number of elements in arrays f and xFreq is identical to
    floor(nT/2) + 1 (nT: the number of elements in arrays t and xTime). This means that the the Fourier coefficients
    are computed up to the Nyquist frequency. Transforming the signal xFreq = xFreq(f) back to time domain can be
    done using first the function unfoldspectrum and after that the function freq2time. Parameter isOdd=1 if nT is
    odd, and isOdd=0 if nT is even.

    Parameters
    ----------
    t : NDArray[float64]
        1D array containing time axis. The numbers in the array t must be increasing and uniformly spaced
        (uniform time step) [s]
    xTime : NDArray[float64]
        1D array containing signal values, i.e. the time series of the signal. The value x(i) must be the signal
        value at time t(i). Usually water surface elevation [m]

    Returns
    -------
    tuple[NDArray[float64], NDArray[complex128], bool]
        f : NDArray[float64]
            1D array containing frequency values, for folded Fourier transform. The frequency axis runs from 0 to the
            Nyquist frequency. The number of elements in array f is close to half the number of elements in array
            xTime. To be precise, length(f) = floor(nT/2) + 1, with nT the number of elements in array xTime [Hz]
        xFreq : NDArray[complex128]
            1D array (complex!) containing the folded Fourier coefficients of xTime. The value xFreq(i) must be the
            Fourier coefficient at frequency f(i). The number of elements in f and xFreq are the same.
        isOdd : bool
            logical indicating whether nT, the number of time points in xTime, is even (isOdd=False) or odd
            (isOdd=True)

    Raises
    ------
    ValueError
        Input error: input should be 1d arrays
    ValueError
        Input error: time input parameter must be monotonic with constant step size
    ValueError
        Input error: array sizes differ in dimension

    Example
    -------
    >>> import numpy as np
    >>> dt        = 0.1 #Hz
    >>> t         = np.arange(0,100)*dt
    >>> a1 = 0.5; w1 = 2*np.pi/5; phi1 = 0.35;  # Wave component 1
    >>> a2 = 0.7; w2 = 2*np.pi/6; phi2 = 0.96;  # Wave component 2
    >>> y = a1*np.cos(w1*t+phi1) + a2*np.cos(w2*t+phi2)
    >>> print(t)
    >>> print(y)
    >>> # --- Compute discrete Fourier transform
    >>> f,yFreq,isOdd = time2freq_nyquist(t,y);

    """
    t, tSize = core_engine.convert_to_vector(t)
    xTime, xTimeSize = core_engine.convert_to_vector(xTime)

    if tSize[1] > 1 or xTimeSize[1] > 1:
        raise ValueError("time2freq_nyquist: Input error: input should be 1d arrays")

    if not core_engine.monotonic_increasing_constant_step(t):
        raise ValueError(
            "time2freq_nyquist: Input error: time input parameter must be monotonic with constant step size"
        )

    if not (tSize[0] == xTimeSize[0]):
        raise ValueError(
            "time2freq_nyquist: Input error: array sizes differ in dimension"
        )

    # Computational core Transform time signal to frequency domain, over
    # frequency axis up to twice the Nyquist frequency
    fTotal, xFreqTotal = time2freq(t, xTime)

    # --- Number of time points, and check whether this number is even or odd
    nT = tSize[0]
    isOdd = bool(nT % 2)

    # --- Index in array that corresponds to the Nyquist frequency
    #     Nyquist frequency = 1 / (2*dt)
    iNyq = math.floor(nT / 2) + 1

    # --- Take part of the signal up to the Nyquist frequency
    f = fTotal[0:iNyq]
    xFreq = xFreqTotal[0:iNyq]

    #
    return f, xFreq, isOdd


def compute_spectrum_time_series(
    t: NDArray[float64],
    xTime: NDArray[float64],
    dfDesired: float = 0.0,
) -> tuple[NDArray[float64], NDArray[float64]]:
    """Computes variance density spectrum from given time series

    This function computes a variance density spectrum sVarDens_coarse = sVarDens_coarse(f_coarse) on a frequency axis
    f_coarse from a given surface elevation time series xTime = xTime(t), with time axis t. The frequency spacing of
    the output spectrum is given by dfDesired.

    Parameters
    ----------
    t : NDArray[float64]
        1D array containing time axis. The numbers in the array t must be increasing and uniformly spaced
        (uniform time step) [s]
    xTime : NDArray[float64]
        1D array containing signal values, i.e. the time series of the signal. The value xTime(i) must be the signal
        value at time t(i). Usually water surface elevation [m]
    dfDesired : float, optional
        desired frequency spacing in Hertz on which the wave spectrum must be computed. If this parameter is omitted,
        then dfDesired = f(1) - f(0), by default 0.0 [Hz]

    Returns
    -------
    tuple[NDArray[float64], NDArray[float64]]
        f_coarse : NDArray[float64]
            frequency axis of computed spectrum. The frequency spacing is (close to) dfDesired [Hz]
        sVarDens_coarse : NDArray[float64]
            band averaged variance density spectrum of the signal on new frequency axis fCoarse [m^2/Hz]

    Example
    -------
    >>> import numpy as np
    >>> dt =0.1
    >>> t = np.arange(0,1000+dt,dt)  # Time axis
    >>> z = np.sin(t) + np.cos(2*t)  # Surface elevation data
    >>> df = 0.01                    # Choose value for frequency axis
    >>> [freq,varDens] = compute_spectrum_time_series(t,z,df)

    """
    # --- Transform to frequency domain ( input check is done in time2freq_nyquist)
    [f, xFreq, _] = time2freq_nyquist(t, xTime)
    df = f[1] - f[0]
    Ntime = len(t)
    sFine = 2 * xFreq * np.conj(xFreq) / (df * Ntime * Ntime)
    sFine = sFine.real

    # --- Perform averaging
    [f_coarse, sVarDens_coarse] = frequency_averaging(f, sFine, dfDesired)

    return f_coarse, sVarDens_coarse


def compute_spectrum_freq_series(
    f: NDArray[float64],
    xFreq: NDArray[float64],
    Ntime: int,
    dfDesired: float = 0.0,
) -> tuple[NDArray[float64], NDArray[float64]]:
    """Computes variance density spectrum from given complex spectrum of Fourier components

    This function computes a variance density spectrum sVarDens_coarse = sVarDens_coarse(f_coarse) on a frequency axis
    f_coarse from a given complex spectrum xFreq = xFreq(f) of Fourier coefficients on a frequency axis f. The
    frequency spacing of the output spectrum is given by dfDesired.

    Parameters
    ----------
    f : NDArray[float64]
        1D array containing frequency values, for folded Fourier transform. The number of elements in array f is
        close to half the number of elements in array xTime. To be precise, length(f) = floor(nT/2) + 1, with nT the
        number of elements in array xTime [Hz]
    xFreq : NDArray[float64]
        wave spectrum of complex Fourier coefficients
    dfDesired : float, optional
        desired frequency spacing in Hertz on which the wave spectrum must be computed. If this parameter is omitted,
        then dfDesired = f(1) - f(0), by default 0.0 [Hz]
    Ntime : int, optional
        number of time elements in original time signal, by default 0

    Returns
    -------
    tuple[NDArray[float64], NDArray[float64]]
        f_coarse : NDArray[float64]
            frequency axis of computed spectrum. The frequency spacing is (close to) dfDesired [Hz]
        sVarDens_coarse : NDArray[float64]
            band averaged variance density spectrum of the signal on new frequency axis fCoarse [m^2/Hz]

    Raises
    ------
    ValueError
        Input error: input should be 1d arrays
    ValueError
        Input error: frequency input parameter must be monotonic with constant step size
    ValueError
        Input error: array sizes differ in dimension

    """
    f, fSize = core_engine.convert_to_vector(f)
    xFreq, xFreqSize = core_engine.convert_to_vector(xFreq)

    if fSize[1] > 1 or xFreqSize[1] > 1:
        raise ValueError(
            "compute_spectrum_freq_series: Input error: input should be 1d arrays"
        )

    if not core_engine.monotonic_increasing_constant_step(f):
        raise ValueError(
            "compute_spectrum_freq_series: Input error: frequency input parameter must be monotonic with constant step"
            " size"
        )

    if not (fSize[0] == xFreqSize[0]):
        raise ValueError(
            "compute_spectrum_freq_series: Input error: array sizes differ in dimension"
        )

    # --- Transform to frequency domain
    df = f[1] - f[0]
    sFine = np.real(2 * xFreq * np.conj(xFreq) / (df * Ntime * Ntime))

    # --- Perform averaging
    [f_coarse, sVarDens_coarse] = frequency_averaging(f, sFine, dfDesired)

    return f_coarse, sVarDens_coarse


def spectrum2timeseries(
    f: NDArray[float64],
    sVarDens: NDArray[float64],
    tInit: float,
    tEnd: float,
    dt: float,
    seed: int = -1,
) -> tuple[NDArray[float64], NDArray[float64]]:
    """Generates a timeseries based on a given spectrum.

    This function synthesises a time series xTime = xTime(t) satisfying a given 1D variance density spectrum sVarDens
    = sVarDens(f). The time axis is specified by the user as well, starting at time tInit, ending at time tEnd, with
    a uniform time step equal to dt. The random phase method is used to synthetise the time series.

    Parameters
    ----------
    f : NDArray[float64]
        1D array containing frequency values, for folded Fourier transform. The number of elements in array f is
        close to half the number of elements in array xTime. To be precise, length(f) = floor(nT/2) + 1, with nT the
        number of elements in array xTime [Hz]
    sVarDens : NDArray[float64]
        1D array containing variance density spectrum of the signal [m^2/Hz]
    tInit : float
        initial time of time axis. [s]
    tEnd : float
        end time of time axis. [s]
    dt : float
        time step of time axis. [s]
    seed : int, optional
        Seed for random phases, by default -1

    Returns
    -------
    tuple[NDArray[float64], NDArray[float64]]
        t : NDArray[float64]
            time axis of computed spectrum. The time spacing is (close to) dt [s]
        xTime : NDArray[float64]
            time series of water level elevation [m]

    Raises
    ------
    ValueError
        Input error: input should be 1d arrays
    ValueError
        Input error: array sizes differ in dimension

    """
    f, fSize = core_engine.convert_to_vector(f)
    xFreq, xFreqSize = core_engine.convert_to_vector(sVarDens)

    if fSize[1] > 1 or xFreqSize[1] > 1:
        raise ValueError(
            "compute_spectrum_freq_series: Input error: input should be 1d arrays"
        )

    if not (fSize[0] == xFreqSize[0]):
        raise ValueError(
            "compute_spectrum_freq_series: Input error: array sizes differ in dimension"
        )

    t = np.arange(tInit, tEnd + dt, dt)

    nTime = len(t)
    nTimeEven = int(2 * np.ceil(nTime / 2))
    tDuration = nTimeEven * dt

    # --- Generate frequency axis fGen, with lowest frequency equal to df and
    #     highest frequency equal to the Nyquist frequency fNyq.
    #     Note that the the number of frequency points nF is equal to
    #     (nTimeEven / 2)
    dfGen = 1 / tDuration
    fNyq = 1 / (2 * dt)
    fGen = np.arange(dfGen, fNyq + dfGen / 2, dfGen)  # TODO why dfGen/2
    nF = len(fGen)

    # --- Interpolate the given variance density spectrum sVarDens = sVarDens(f)
    #     to a spectrum sVarDensGen = sVarDensGen(fGen), that is, at
    #     frequencies fGen.
    #     The values of sVarDensGen are put to zero outside the originally
    #     given range of f
    sVarDensGen = np.interp(fGen, f, sVarDens, left=0, right=0)

    # --- Chose nF random phases and, if requested, reset the state.
    #     Resetting the state means that a renewed call to the present function
    #     with the same arguments will give the same random numbers
    if seed >= 0:
        np.random.seed(seed)

    phase = 2 * np.pi * np.random.rand(1, nF)

    # --- Determine amplitudes per frequency band, based on the method of Miles
    amplitude = np.sqrt(2 * sVarDensGen * dfGen)

    # --- Construct the Fourier components

    xFreq = np.zeros(nTimeEven, dtype=complex128)
    xFreq[1 : nF + 1] = amplitude / 2.0 * np.exp(1j * phase)
    xFreq[nF:nTimeEven] = np.flip(np.conj(xFreq[1 : nF + 1]))

    # --- Inverse Fourier transformation to time domain
    xTime = np.real(np.fft.ifft(xFreq, nTimeEven) * nTimeEven)

    # --- Take the part corresponding to the time axis
    xTime = xTime[0:nTime]
    return t, xTime


def spectrum2timeseries_object(
    f: NDArray[float64],
    sVarDens: NDArray[float64],
    tInit: float,
    tEnd: float,
    dt: float,
    seed: int = -1,
) -> series.Series:
    """Generates a timeseries based on a given spectrum.

    This function synthesises a time series xTime = xTime(t) satisfying a given 1D variance density spectrum sVarDens
    = sVarDens(f). The time axis is specified by the user as well, starting at time tInit, ending at time tEnd, with
    a uniform time step equal to dt. The random phase method is used to synthetise the time series.

    Parameters
    ----------
    f : NDArray[float64]
        1D array containing frequency values, for folded Fourier transform. The number of elements in array f is
        close to half the number of elements in array xTime. To be precise, length(f) = floor(nT/2) + 1, with nT the
        number of elements in array xTime [Hz]
    sVarDens : NDArray[float64]
        1D array containing variance density spectrum of the signal [m^2/Hz]
    tInit : float
        initial time of time axis. [s]
    tEnd : float
        end time of time axis. [s]
    dt : float
        time step of time axis. [s]
    seed : int, optional
        Seed for random phases, by default -1

    Returns
    -------
    series.Series
        Series object containing the time series of water level elevation

    """
    t, xTime = spectrum2timeseries(f, sVarDens, tInit, tEnd, dt, seed)
    return series.Series(t, xTime)


def compute_spectrum_welch_wrapper(
    xTime: NDArray[float64],
    dt: float,
    nperseg: int = 0,
    noverlap: int = 0,
    nfft: int = 0,
    window_type: str = "hann",
) -> tuple[NDArray[float64], NDArray[float64]]:
    """Wrapper arround the Scipy Welch method

    Parameters
    ----------
    xTime : NDArray[float64]
        1D array containing signal values, i.e. the time series of the signal. The value xTime(i) must be the signal
        value at time t(i). Usually water surface elevation [m]
    dt : float
        time step of signal
    nperseg : int, optional
        Length of each segment, by default None
    noverlap : int, optional
        number of points in overlap, by default None
    nfft : int, optional
        length of fft, by default None
    window_type : str, optional
        window type, by default "hann"

    Returns
    -------
    tuple[NDArray[float64], NDArray[float64]]
        f : NDArray[float64]
            1D array containing frequency values, for folded Fourier transform. The number of elements in array f is
            close to half the number of elements in array xTime. To be precise, length(f) = floor(nT/2) + 1, with nT
            the number of elements in array xTime [Hz]
        sVarDens : NDArray[float64]
            1D array containing variance density spectrum of the signal [m^2/Hz]

    """
    if nperseg == 0:
        nperseg = None
    if noverlap == 0:
        noverlap = None
    if nfft == 0:
        nfft = None
    f, sVarDens = welch(
        xTime,
        fs=1 / dt,
        window=window_type,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend="constant",
        return_onesided=True,
        scaling="density",
        axis=-1,
        average="mean",
    )
    return f, sVarDens


def applybandfilter(
    f: NDArray[float64], xFreq: NDArray[float64], flow: float, fhigh: float
):
    """applybandfilter

    Parameters
    ----------
    f : NDArray[float64]
        1D array containing the frequencies
    xFreq : NDArray[float64]
        1D array (complex!) containing the folded Fourier coefficients of xTime. The value xFreq(i) must be the
        Fourier coefficient at frequency f(i). The number of elements in f and xFreq are the same.
    flow : float
        Lower frequeny of the bandpassfilter
    fhigh : float
        Higher frequency of the bandpassfilter


    Returns
    -------
    tuple NDArray[float64]]
        zFreq : NDArray[float64]
            1D array containing the folded Fourier coefficients.

    """
    # --- Length of array f = length of array xFreq
    Nf = len(f)

    # --- Find index in frequency array corresponding to flow
    if flow <= f[0]:
        ilow = 1
    elif flow >= f[-1]:
        ilow = Nf
    else:
        ilow = np.where(f < flow)[0][-1] + 1

    # --- Find index in frequency array corresponding to fhigh
    if fhigh <= f[0]:
        ihigh = 1
    elif fhigh >= f[-1]:
        ihigh = Nf
    else:
        ihigh = np.where(f < fhigh)[0][-1]

    # --- Initialize
    zFreq = np.zeros(len(xFreq)).astype(complex)

    # --- Apply band filter
    zFreq[ilow:ihigh] = xFreq[ilow:ihigh]
    return zFreq


def bandpassfilter(
    t: NDArray[float64], xTime: NDArray[float64], flow: float, fhigh: float
):
    """Bandpassfilter
    Filters a signal based on a lower and higher frequency limit.

    Parameters
    ----------
    t : NDArray[float64]
        1D array containing the time
    xTime : NDArray[float64]
        1D array containing signal values, i.e. the time series of the signal. The value xTime(i) must be the signal
        value at time t(i). Usually water surface elevation [m]
    flow : float
        Lower frequeny of the bandpassfilter
    fhigh : float
        Higher frequency of the bandpassfilter


    Returns
    -------
    tuple NDArray[float64]]
        zTime : NDArray[float64]
            1D array containing the filtered signal.

    """
    [f, xFreq, isOdd] = time2freq_nyquist(t, xTime)

    zFreq = applybandfilter(f, xFreq, flow, fhigh)
    [fUnfold, zFreqUnfold] = unfold_spectrum(f, zFreq, isOdd)
    zTime = freq2time(zFreqUnfold)
    return zTime
