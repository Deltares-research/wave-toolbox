# SPDX-License-Identifier: GPL-3.0-or-later
import copy

import numpy as np
from numpy import float64
from numpy.typing import NDArray

import deltares_wave_toolbox.cores.core_dispersion as core_dispersion
import deltares_wave_toolbox.cores.core_spectral as core_spectral
import deltares_wave_toolbox.series as series


def decompose_linear_ZS(
    t: NDArray[float64],
    xTime: NDArray[float64],
    h: NDArray[float64],
    x_loc: NDArray[float64],
    w_loc: NDArray[float64],
    detLim: float = 0.125,
) -> tuple[NDArray[float64], NDArray[float64]]:
    """
    decompose_linear_ZS  Separation of incident and reflected waves using Zelt & Skjelbreia's method.

    This function performs the separation of incident and reflected waves
    according to the Zelt & Skjelbreia's procedure. Time series of wave
    signals xTime(Ntime,Nloc) at locations x_loc(Nloc) need to be provided.
    All wave gauges are assigned a weight w_loc(Nloc).
    Waves can only propagate in the positive x-direction (incoming wave) and
    the negative x-direction (reflected wave). The water depth h is assumed
    to be uniform. Time series of the incident and reflected wave in
    location x_loc(1) are provided as output of this function.

    For Nloc = 2 and w_loc = [1 1], this reduces to Goda's and Suzuki's
    method.
    For Nloc = 3 and w_loc = [1 1 1], this reduces to Mansard-Funke method.

    Parameters
    ----------
    t      :
           1D real array containing Ntime time values. The numbers in the
           array t must be increasing and uniformly spaced (uniform time
           step). The initial time t(1) can be any value (so it is not
           obligatory to have t(1) = 0). Units: seconds
    xTime  :
           real array of size [Ntime,Nloc] containing the time series of
           the surface elevation in locations x_loc.
           Units: meters
    h      :
           depth, in meters. Assumed to be uniform
    x_loc  :
           1D real array with Nloc entries indicating the locations where
           the time signals are measured.
    w_loc  :
           1D real array with Nloc entries indicating the weights of the
           signals. The values of w_loc must be between zero and 1.
    detLim :
           Optional parameter, with default value 0.125.
           This parameter, which is essentially borrowed from the
           Mansard-Funke function, indicates the value of the matrix
           determinantabove which the Mansard-Funke procedure is applied.
           When thedeterminant is below the value detLim, a more crude
           splitting procedure is applied. The value of detLim must be
           small and positive. Parameter detLim is related to the
           parameter denominator in the Auke-Process input. The relation
           reads:
               denominator = 9 - 4 * detLim.
           Hence, for detLim = 0.125, we get denominator = 8.5 (default
           value in Auke-Process).

    Returns
    -------
    xTime1In :
             1D real array containing time signal of incident wave (i.e.
             propagating in positive x-direction), in location x_loc(1)
    xTime1Re :
             1D real array containing time signal of reflected wave (i.e.
             propagating in negative x-direction), in location x_loc(1)

    Remark:
       * The wave signals must be ordered such that x_loc(1) < x_loc(2) < ....
       * Nloc must be two or larger
       * The sum of the weights w_loc must be two or larger

    Syntax:
       xTime1In,xTime1Re = decompose_linear_ZS(t,xTime,h,x_loc,w_loc,detLim)

    """

    # --- Some checks on input

    # --- Check whether input arrays are consistent in the sense that they
    #     share the same number of locations
    Ntime = 0
    Nloc = 0
    Ntime, Nloc = xTime.shape
    Nloc2 = len(x_loc)
    Nloc3 = len(w_loc)

    assert (
        Nloc == Nloc2
    ), "ZS: input error: Dimensions of xTime and x_loc do not correspond."
    assert (
        Nloc2 == Nloc3
    ), "ZS: input error: Dimensions of x_loc and w_loc do not correspond."
    assert Nloc > 1, "ZS: input error: Number of wave gauges must be at least 2."

    # --- Check whether the number of time samples agrees
    assert Ntime == len(
        t
    ), "ZS: input error: Dimensions of x_loc and t do not correspond."
    # --- Values of w_loc should be between 0 and 1
    assert (
        min(w_loc) > 0 or max(w_loc) <= 1
    ), "ZS: input error: Values of w_loc must be between 0 and 1."

    w_loc_sum = sum(w_loc)
    assert (
        w_loc_sum > 1
    ), "ZS: input error: Sum of elements in w_loc must be at least 2."

    # --- Check whether x_loc(1) < x_loc(2) < ...
    assert (
        min(np.diff(x_loc)) > 0
    ), 'ZS: input error": Wave gauges positions x_loc are not sorted in increasing order.'

    # --- Computational core
    # --- Some constants
    g = 9.81

    #
    diff_x_loc = np.zeros(Nloc)

    for iloc in np.arange(1, Nloc):
        diff_x_loc[iloc] = x_loc[iloc] - x_loc[0]

    # --- Compute Fourier transform of the time signals
    f, xFreq1, isOdd = core_spectral.time2freq_nyquist(t, xTime[:, 0])
    Nf = len(f)

    # to solve waring: ComplexWarning: Casting complex values to real discards the imaginary part.
    # note 2d dimension is specified as one argument (Nf,Nloc) for function zeros()
    xFreq = np.zeros((Nf, Nloc), dtype=complex)
    xFreq[:, 0] = copy.deepcopy(xFreq1)
    # note iloc runs from 1 to Nloc-1 when using range(1,Nloc)
    for iloc in np.arange(1, Nloc):
        f, xFreq[:, iloc], isOdd = core_spectral.time2freq_nyquist(t, xTime[:, iloc])

    # --- Compute the wave number
    w = 2 * np.pi * f
    k = core_dispersion.disper(w, h, g)

    # --- Initialize complex spectrum of incident and reflected waves
    #     Their initial value is, as is done also in Auke-Process, set to zero.
    #     For most frequencies (i.e. the ones corresponding to a matrix
    #     determinant larger than detLim), the values as computed from the
    #     Mansard-Funke procedure are inserted in xFreq1In and xFreq1Re. At
    #     the frequencies with a determinant smaller than detLim (these are the
    #     smallest frequencies), the zero initialisation means that all energy
    #     at these frequencies is discarded. In the future perhaps a better
    #     initialisation approach must be found, such as for example:
    #        xFreq1In = (1-r)*xFreq1; xFreq1Re = r*xFreq1;
    #     where r is some user-defined parameter or a measure for the
    #     reflection.
    xFreq1In = np.zeros(len(xFreq1), dtype=complex)
    xFreq1Re = np.zeros(len(xFreq1), dtype=complex)

    # --- Loop over all frequencies
    a12 = w_loc_sum
    a21 = w_loc_sum
    for ifreq in np.arange(0, Nf):
        # --- Initialize matrix elements and right-hand sides
        a11 = 0
        a22 = 0
        b1 = 0
        b2 = 0
        # --- Compute matrix elements and right-hand sides
        for iloc in np.arange(0, Nloc):
            a11 = a11 + w_loc[iloc] * np.exp(-1j * 2 * k[ifreq] * diff_x_loc[iloc])
            a22 = a22 + w_loc[iloc] * np.exp(1j * 2 * k[ifreq] * diff_x_loc[iloc])
            #
            b1 = b1 + w_loc[iloc] * xFreq[ifreq, iloc] * np.exp(
                -1j * k[ifreq] * diff_x_loc[iloc]
            )
            b2 = b2 + w_loc[iloc] * xFreq[ifreq, iloc] * np.exp(
                1j * k[ifreq] * diff_x_loc[iloc]
            )

        # --- Compute determinant of matrix (for all frequency components)
        detA = a11 * a22 - a12 * a21

        # --- The scaled determinant sDetA is equal to:
        sDetA = abs(-0.25 * detA)

        # --- Consider only frequency components for which sDetA > detLim. For
        #     these frequency components, solve 2x2 system analytically
        if sDetA > detLim:
            xFreq1In[ifreq] = (a22 * b1 - a12 * b2) / detA
            xFreq1Re[ifreq] = (-a21 * b1 + a11 * b2) / detA

    # --- Unfold spectra
    fTot, xFreq1InTot = core_spectral.unfold_spectrum(
        f.reshape(len(f), 1), xFreq1In.reshape(len(f), 1), isOdd
    )
    fTot, xFreq1ReTot = core_spectral.unfold_spectrum(
        f.reshape(len(f), 1), xFreq1Re.reshape(len(f), 1), isOdd
    )

    # --- Transform back to time domain
    xTime1In = core_spectral.freq2time(xFreq1InTot)
    xTime1Re = core_spectral.freq2time(xFreq1ReTot)

    return xTime1In, xTime1Re


def decompose_linear_ZS_series(
    Series_objects: list[series.Series],
    h: NDArray[float64],
    x_loc: NDArray[float64],
    w_loc: NDArray[float64],
    detLim: float = 0.125,
):
    """Separation of incident and reflected waves using the Zelt & Skjelbreia (1992) method.

    This is a wrapper function for the decompose_linear_ZS function, made compatible with
    the Series class from the Deltares Wave Toolbox.

    This function performs the separation of incident and reflected waves
    according to the Zelt & Skjelbreia (1992) procedure. Time series of wave
    signals xTime(Ntime,Nloc) at locations x_loc(Nloc) need to be provided.
    All wave gauges are assigned a weight w_loc(Nloc).
    Waves can only propagate in the positive x-direction (incoming wave) and
    the negative x-direction (reflected wave). The water depth h is assumed
    to be uniform. Time series of the incident and reflected wave in
    location x_loc(1) are provided as output of this function.

    For Nloc = 2 and w_loc = [1 1], this reduces to Goda's and Suzuki's
    method.
    For Nloc = 3 and w_loc = [1 1 1], this reduces to Mansard-Funke method.

    Parameters
    ----------
    Series_objects : list[series.Series]
        List of Series objects containing the time series of the surface elevation in locations x_loc.
    h : NDArray[float64]
        depth, in meters. Assumed to be uniform
    x_loc : NDArray[float64]
        1D real array with Nloc entries indicating the locations where
        the time signals are measured.
    w_loc : NDArray[float64]
        1D real array with Nloc entries indicating the weights of the
        signals. The values of w_loc must be between zero and 1.
    detLim : float, optional
        Optional parameter, by default 0.125
           This parameter, which is essentially borrowed from the
           Mansard-Funke function, indicates the value of the matrix
           determinantabove which the Mansard-Funke procedure is applied.
           When thedeterminant is below the value detLim, a more crude
           splitting procedure is applied. The value of detLim must be
           small and positive. Parameter detLim is related to the
           parameter denominator in the Auke-Process input. The relation
           reads:
               denominator = 9 - 4 * detLim

    Returns
    -------
    xTime1In :
             1D real array containing time signal of incident wave (i.e.
             propagating in positive x-direction), in location x_loc(1)
    xTime1Re :
             1D real array containing time signal of reflected wave (i.e.
             propagating in negative x-direction), in location x_loc(1)
    """

    for i, series_object in enumerate(Series_objects):

        if i == 0:
            t = series_object.time
            xTime = np.zeros((len(t), len(x_loc)))

        xTime[:, i] = series_object.xTime

    xTime1In, xTime1Re = decompose_linear_ZS(t, xTime, h, x_loc, w_loc, detLim)

    xTime1In_series = series.Series(t, xTime1In)
    xTime1Re_series = series.Series(t, xTime1Re)

    return xTime1In_series, xTime1Re_series
