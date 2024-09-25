# SPDX-License-Identifier: GPL-3.0-or-later
import warnings

import numpy as np
import scipy.integrate as integrate
import scipy.special as special
from numpy import float64
from numpy.typing import NDArray
from scipy.stats import exponweib

import deltares_wave_toolbox.cores.core_engine as core_engine
import deltares_wave_toolbox.spectrum as spectrum


def compute_spectrum_params(
    f: NDArray[float64],
    sVarDens: NDArray[float64],
    fmin: float = -1.0,
    fmax: float = -1.0,
) -> tuple[float, float, float, float, float, float]:
    """Computes spectral parameters of given spectrum

    This function computes several spectral wave parameters of a given 1D
    spectrum

    Parameters
    ----------
    f : NDArray[float64]
        1D array containing frequency values. The numbers in the array f must be increasing and uniformly spaced
        (uniform frequency step). [Hz]
    sVarDens : NDArray[float64]
        1D array containing variance density spectrum of the signal [m^2/Hz]
    fmin : float, optional
        lower bound of the moment integral [Hz], by default -1.0
    fmax : float, optional
        upper bound of the moment integral [Hz], by default -1.0

    Returns
    -------
    tuple[float, float, float, float, float, float]
        Hm0 : float
            spectral wave height [m]
        Tp : float
            peak period [s]
        Tps : float
            smoothed peak period [s]
        Tmm10 : float
            wave period based on (-1) and (0) moments [s]
        Tm01 : float
            wave period based on (0) and (1) moments [s]
        Tm02 : float
            wave period based on (0) and (2) moments [s]

    Raises
    ------
    ValueError
        Input error: input should be 1d arrays
    ValueError
        Input error: frequency input parameter must be monotonic with constant step size
    ValueError
        Input error: array sizes differ in dimension

    Example
    -------
    >>> import numpy as np
    >>> f = np.arange(0,1,0.1)
    >>> Tp = 5.0
    >>> Hm0 = 1.0
    >>> fmin =0.01
    >>> fmax =1.0
    >>> sPM = create_spectrum_piersonmoskowitz(f,1/Tp,Hm0)
    >>> [Hm0,Tp,Tps,Tmm10,Tm01,Tm02] = compute_spectrum_params(f,sPM,fmin,fmax)

    """
    # --- Ensure array input is of type ndarray.
    f, fSize = core_engine.convert_to_vector(f)
    sVarDens, SSize = core_engine.convert_to_vector(sVarDens)

    if fSize[1] > 1 or SSize[1] > 1:
        raise ValueError(
            "compute_spectrum_params: Input error: input should be 1d arrays"
        )

    if not core_engine.monotonic_increasing_constant_step(f):
        raise ValueError(
            "compute_spectrum_params: Input error: frequency input parameter must be monotonic with constant step size"
        )

    if not (fSize[0] == SSize[0]):
        raise ValueError(
            "compute_spectrum_params: Input error: array sizes differ in dimension"
        )

    #
    # --- Find values of fmin and fmax
    if fmin == -1.0:
        fmin = f[0]
    if fmax == -1.0:
        fmax = f[fSize[0] - 1]

    # --- Compute moments
    m_1 = compute_moment(f, sVarDens, -1, fmin, fmax)
    m0 = compute_moment(f, sVarDens, 0, fmin, fmax)
    m1 = compute_moment(f, sVarDens, 1, fmin, fmax)
    m2 = compute_moment(f, sVarDens, 2, fmin, fmax)

    # --- Compute wave height -----------------------------------------------
    Hm0 = float(4 * np.sqrt(m0))

    # --- Put values to NaN (exception value) in situation that wave height is
    #     (virtually) zero
    if Hm0 < 1e-6 or np.isnan(Hm0):
        Hm0 = np.nan
        Tp = np.nan
        Tps = np.nan
        Tmm10 = np.nan
        Tm01 = np.nan
        Tm02 = np.nan
        return Hm0, Tp, Tps, Tmm10, Tm01, Tm02

    # --- Compute mean wave periods -----------------------------------------
    Tmm10 = m_1 / m0
    Tm01 = m0 / m1
    Tm02 = float(np.sqrt(m0 / m2))

    # --- Make separate arrays containing only part corresponding to
    #     frequencies between fmin and fmax
    iFmin = np.where(f >= fmin)[0][0]
    iFmax = np.where(f <= fmax)[0][-1]
    fMiMa = f[iFmin : iFmax + 1]
    SMiMa = sVarDens[iFmin : iFmax + 1]

    # --- Compute peak period -----------------------------------------------
    Smax = max(SMiMa)
    imax = np.where(SMiMa == Smax)[0]
    imax = imax.astype(int)
    ifp = max(imax)
    fp = fMiMa[ifp]

    if np.all(ifp is None):
        ifp = 1
        fp = fMiMa[ifp]
    Tp = float(1 / fp)

    # --- Compute smoothed peak period --------------------------------------
    Tps = compute_tps(fMiMa, SMiMa)
    return Hm0, Tp, Tps, Tmm10, Tm01, Tm02


def compute_moment(
    f: NDArray[float64],
    sVarDens: NDArray[float64],
    m: int,
    fmin: float = -1.0,
    fmax: float = -1.0,
) -> float:
    """Computes the spectral moment

    This function computes the m'th order spectral moment of variance density spectrum S=S(f), with f the frequency
    axis, over frequency domain [fmin,fmax].

    It is required that fmin >= f_in(1).

    It is not required to have fmax <= f_in(end). So it ok to have fmax = Inf.
    If fmax>f(end), then the moment consists of the summation of two parts:
    (1) Integration of (f_in^m * S), with given S, over [fmin,f(end)]
    (2) Exact integration of (f^m * S_lim) over [f(end),fmax], where S_lim is a high-frequency f^(-5) tail.
    Typically, in such cases one puts fmax = Inf.

    Parameters
    ----------
    f : NDArray[float64]
        1D array containing frequency values. The numbers in the array f must be increasing and uniformly spaced
        (uniform frequency step). [Hz]
    sVarDens : NDArray[float64]
        1D array containing variance density spectrum of the signal [m^2/Hz]
    m : int
        order of moment
    fmin : float, optional
        lower bound of the moment integral [Hz], by default -1.0
    fmax : float, optional
        upper bound of the moment integral [Hz], by default -1.0

    Returns
    -------
    float
        the computed moment

    Raises
    ------
    ValueError
        Input error: input should be 1d arrays
    ValueError
        Input error: frequency input parameter must be monotonic with constant step size
    ValueError
        Input error: array sizes differ in dimension

    Example
    -------
    >>> import numpy as np
    >>> f = np.arange(0,1,0.1)
    >>> Tp = 5.0
    >>> Hm0 = 1.0
    >>> fmin =0.01
    >>> fmax =1
    >>> sPM = create_spectrum_piersonmoskowitz(f,1/Tp,Hm0)
    >>> m_1 = compute_moment(f,sPM,-1)
    >>> m0  = compute_moment(f,sPM,0)
    >>> m1  = compute_moment(f,sPM,1)
    >>> m2  = compute_moment(f,sPM,2)

    """
    # --- Ensure array input is of type ndarray.
    f, fSize = core_engine.convert_to_vector(f)
    sVarDens, SSize = core_engine.convert_to_vector(sVarDens)

    if fSize[1] > 1 or SSize[1] > 1:
        raise ValueError("compute_moment: Input error: input should be 1d arrays")

    if not core_engine.monotonic_increasing_constant_step(f):
        raise ValueError(
            "compute_moment: Input error: frequency input parameter must be monotonic with constant step size"
        )

    if not (fSize[0] == SSize[0]):
        raise ValueError("compute_moment: Input error: array sizes differ in dimension")

    # --- Remove the possible situation that f=0 in combination with m<0. This
    #     would lead to division by zero
    if m < 0 and f[0] == 0:
        freq = f[1:]
        spec = sVarDens[1:]
    else:
        freq = f
        spec = sVarDens

    # --- Compute the integrand, that is the product of f^m * S
    integrand = freq ** (m) * spec

    # --- Depending on number of input arguments, compute the moment integral
    if fmin == -1.0 or fmax == -1.0:  # integrate over all values in freq interval.
        moment = integrate.simpson(integrand, freq)
    else:  # integrate over all values in sub interval.
        # fmin and fmax are given
        fminn = fmin
        if m < 0 and f[0] == 0:
            if fmin == 0:
                fminn = f[1]

        #
        if fmax <= freq[len(freq) - 1]:
            ifminn = core_engine.approx_array_index(freq, fminn)
            ifmax = core_engine.approx_array_index(freq, fmax) + 1
            moment = integrate.simpson(integrand[ifminn:ifmax], freq[ifminn:ifmax])

        else:
            # 1: Integral over [fminn,freq(end)]
            ifminn = core_engine.approx_array_index(freq, fminn)
            ifmax = core_engine.approx_array_index(freq, freq[len(freq) - 1]) + 1
            moment1 = integrate.simpson(integrand[ifminn:ifmax], freq[ifminn:ifmax])
            # 2: Integral over [freq(end),fmax]
            #    Variance density spectrum in this range: C * f^power, with
            #    C determined by power and spec(end)
            power = -5  # Power of high-frequency tail
            C = spec[len(spec) - 1] / (freq[len(freq) - 1] ** power)
            moment2 = (C / (m + power + 1)) * (
                fmax ** (m + power + 1) - freq[len(freq) - 1] ** (m + power + 1)
            )
            # Add the two moments
            moment = moment1 + moment2
    return moment


def create_spectrum_jonswap(
    f: NDArray[float64],
    fp: float,
    hm0: float,
    gammaPeak: float = 3.3,
    l_fmax: float = 0.0,
) -> NDArray[float64]:
    """Creates a Jonswap spectrum

    This function creates the Jonswap variance density spectrum, based on a given frequency axis, wave height, peak
    frequency and peak enhancement factor.
    Literature:
    Hasselman, K., e.a. (1973), Erga"nzungsheft zur Deutschen
    Hydrographischen Zeitschrift, Reihe A(8), No. 12

    For l_fmax = 0, the sVarDens is such that integration from f(1) to f(end) leads exactly to the given Hm0. For
    l_fmax = 1, integration from [f(end},inf] is computed using a (-5)-power law. This also means that integration
    from f(1) to f(end) leads to a slightly smaller value for the wave height than the prescribed Hm0.

    Parameters
    ----------
    f : NDArray[float64]
        1D array containing frequency values. The numbers in the array f must be increasing and uniformly spaced
        (uniform frequency step). [Hz]
    fp : float
        peak frequency. [Hz]
    hm0 : float
        spectral wave height [m]
    gammaPeak : float, optional
        peak enhancement factor, by default 3.3
    l_fmax : float, optional
        The imposed spectral wave height Hm0 holds for the frequency range [f(1),f(end)] (l_fmax = 0, default) or for
        the frequency range [f(1),inf] (l_fmax = 1)., by default 0

    Returns
    -------
    NDArray[float64]
        1D array containing variance density [m^2/Hz]

    Raises
    ------
    ValueError
        Input error: Input array f is not 1D
    ValueError
        Input error:Argument l_fmax must be either 0 or 1

    Example
    -------
    >>> import numpy as np
    >>> f=np.arange(0,2,0.1)
    >>> Tp = 5.0
    >>> hm0 =1.5
    >>> S = create_spectrum_jonswap(f,1/Tp,hm0,3.3)

    """
    # --- Ensure array input is of type ndarray.
    f, fSize = core_engine.convert_to_vector(f)

    nf = fSize[0]

    # Perform check on input arguments
    # --- Check whether input array f is a 1D vector array
    isvalid_size = nf > 0
    if not isvalid_size:
        raise ValueError(
            "create_spectrum_jonswap:Input error: Input array f is not 1D "
        )

    if l_fmax == 0:
        fmax = f[nf - 1]
    elif l_fmax == 1:
        fmax = np.inf
    else:
        raise ValueError(
            "create_spectrum_jonswap:Input error:Argument l_fmax must be either 0 or 1"
        )

    # Computational core
    # --- Some relevant constants
    sigma_a = 0.07  # Parameter in peak enhancement function
    sigma_b = 0.09  # Parameter in peak enhancement function

    # --- Scaling constant C.
    # Note that scaling with Hm0 to obtain correct magnitude of S
    # is done further below. The scaling constant C is included for reasons of
    # consistency with formulations as present in literature. For
    # computational reasons, it is not needed.
    g = 9.81  # Gravity constant
    alpha = 1  # Scaling parameter, taken equal to 1.
    C = alpha * g**2 * (2 * np.pi) ** (-4)  # Scaling constant

    # --- Initialize variance density spectrum
    sVarDens = np.zeros(len(f))

    # --- Evaluate variance density spectrum, for the moment omitting the
    #     weighting
    for iff in np.arange(0, nf):
        f_i = f[iff]

        # --- Consider only f_i > 0.
        #     For f_i <=0, the variance density is kept equal to zero
        if f_i > np.spacing(1):
            # Ratio f/fp
            nu = f[iff] / fp

            # Parameter sigma
            if f_i < fp:
                sigma = sigma_a
            else:
                sigma = sigma_b

            # Peak enhancement function
            A = np.exp(-((nu - 1) ** 2) / (2 * sigma**2))
            lambda_jonswap = gammaPeak**A

            # Variance density
            sVarDens[iff] = (
                C * f_i ** (-5) * np.exp(-1.25 * nu ** (-4)) * lambda_jonswap
            )

    # --- Compute 'wave height' of the not yet correctly scaled variance
    #     density spectrum
    m0 = compute_moment(f, sVarDens, 0, f[0], fmax)
    hm0NonScale = 4 * np.sqrt(m0)

    # --- Perform scaling, to obtain a variance density that has the proper
    #     energy, i.e. corresponding with wave height Hm0
    sVarDens = sVarDens * (hm0 / hm0NonScale) ** 2
    return sVarDens


def create_spectrum_object_jonswap(
    f: NDArray[float64],
    fp: float,
    hm0: float,
    gammaPeak: float = 3.3,
    l_fmax: float = 0.0,
) -> spectrum.Spectrum:
    """Creates a Jonswap spectrum object

    This function creates the Jonswap variance density spectrum, based on a given frequency axis, wave height, peak
    frequency and peak enhancement factor.

    Literature:
    Hasselman, K., e.a. (1973), Erga"nzungsheft zur Deutschen
    Hydrographischen Zeitschrift, Reihe A(8), No. 12

    For l_fmax = 0, the sVarDens is such that integration from f(1) to f(end) leads exactly to the given Hm0.
    For l_fmax = 1, integration from [f(end},inf] is computed using a (-5)-power law. This also means that
    integration from f(1) to f(end) leads to a slightly smaller value for the wave height than the prescribed Hm0.

    Parameters
    ----------
    f : NDArray[float64]
        1D array containing frequency values. The numbers in the array f must be increasing and uniformly spaced
        (uniform frequency step). [Hz]
    fp : float
        peak frequency. [Hz]
    hm0 : float
        spectral wave height [m]
    gammaPeak : float, optional
        peak enhancement factor, by default 3.3
    l_fmax : float, optional
        The imposed spectral wave height Hm0 holds for the frequency range [f(1),f(end)] (l_fmax = 0, default) or for
        the frequency range [f(1),inf] (l_fmax = 1), by default 0

    Returns
    -------
    spectrum.Spectrum
        Spectrum object

    """
    sVarDens = create_spectrum_jonswap(f, fp, hm0, gammaPeak, l_fmax)
    return spectrum.Spectrum(f, sVarDens)


def create_spectrum_piersonmoskowitz(
    f: NDArray[float64],
    fp: float,
    hm0: float,
    gammaPeak: float = 1.0,
    l_fmax: float = 0.0,
) -> NDArray[float64]:
    """Creates a Pierson-Moskowitz spectrum

    This function creates the Pierson-Moskowitz variance density spectrum, based on agiven frequency axis, wave
    height and peak frequency. The Pierson-Moskowitz spectrum is identical to the Jonswap spectrum with a peak
    enhancement factor equal to 1. Furthermore, the Pierson-Moskowitz spectrum, the Bretschneider spectrum and the
    ITTC spectrum are all three identical.

    Literature:
    Pierson, W.J. and L. Moskowitz (1964). A proposed spectral form for fully developed wind seas based on the
    similarity theory of S.A. Kitaigorodskii. Journal of Geophysical Research,Vol. 69, No. 24, pg. 5181 - 5190.

    For l_fmax = 0, the sVarDens is such that integration from f(1) to f(end) leads exactly to the given Hm0.
    For l_fmax = 1, integration from [f(end},inf] is computed using a (-5)-power law. This also means that
    integration from f(1) to f(end) leads to a slightly smaller value for the wave height than the prescribed Hm0.

    Parameters
    ----------
    f : NDArray[float64]
        1D array containing frequency values. The numbers in the array f must be increasing and uniformly spaced
        (uniform frequency step). [Hz]
    fp : float
        peak frequency. [Hz]
    hm0 : float
        spectral wave height [m]
    gammaPeak : float, optional
        peak enhancement factor, by default 1.0
    l_fmax : float, optional
        The imposed spectral wave height Hm0 holds for the frequency range [f(1),f(end)] (l_fmax = 0, default) or for
        the frequency range [f(1),inf] (l_fmax = 1), by default 0

    Returns
    -------
    NDArray[float64]
        1D array containing variance density [m^2/Hz]

    Example
    -------
    >>> import numpy as np
    >>> f=np.arange(0,2,0.01)
    >>> Tp = 5.0
    >>> hm0 =1.5
    >>> Spm = create_spectrum_piersonmoskowitz(f,1/Tp,hm0)

    """
    # --- Ensure array input is of type ndarray.
    f, _ = core_engine.convert_to_vector(f)

    # Computational core
    # --- Use the fact that the Pierson-Moskowitz spectrum is identical to the
    #     Jonswap spectrum with a peak enhancement factor equal to 1.
    return create_spectrum_jonswap(f, fp, hm0, gammaPeak, l_fmax)


def create_spectrum_object_piersonmoskowitz(
    f: NDArray[float64],
    fp: float,
    hm0: float,
    gammaPeak: float = 1.0,
    l_fmax: float = 0,
) -> spectrum.Spectrum:
    """Creates a Pierson-Moskowitz spectrum object

    This function creates the Pierson-Moskowitz variance density spectrum, based on agiven frequency axis, wave
    height and peak frequency. The Pierson-Moskowitz spectrum is identical to the Jonswap spectrum with a peak
    enhancement factor equal to 1. Furthermore, the Pierson-Moskowitz spectrum, the Bretschneider spectrum and the
    ITTC spectrum are all three identical.

    Literature:
    Pierson, W.J. and L. Moskowitz (1964). A proposed spectral form for fully developed wind seas based on the
    similarity theory of S.A. Kitaigorodskii. Journal of Geophysical Research,Vol. 69, No. 24, pg. 5181 - 5190.

    For l_fmax = 0, the sVarDens is such that integration from f(1) to f(end) leads exactly to the given Hm0.
    For l_fmax = 1, integration from [f(end},inf] is computed using a (-5)-power law. This also means that
    integration from f(1) to f(end) leads to a slightly smaller value for the wave height than the prescribed Hm0.

    Parameters
    ----------
    f : NDArray[float64]
        1D array containing frequency values. The numbers in the array f must be increasing and uniformly spaced
        (uniform frequency step). [Hz]
    fp : float
        peak frequency. [Hz]
    hm0 : float
        spectral wave height [m]
    gammaPeak : float, optional
        peak enhancement factor, by default 1.0
    l_fmax : float, optional
        The imposed spectral wave height Hm0 holds for the frequency range [f(1),f(end)] (l_fmax = 0, default) or for
        the frequency range [f(1),inf] (l_fmax = 1), by default 0

    Returns
    -------
    spectrum.Spectrum
        Spectrum object

    """
    return create_spectrum_object_jonswap(f, fp, hm0, gammaPeak, l_fmax)


def tpd(f: NDArray[float64], sVarDens: NDArray[float64]) -> float:
    """Function which calculates the spectral period (s)

    For definition of TpD: transition from peak wave period to spectral period for the design of placed block revetments

    Parameters
    ----------
    f : NDArray[float64]
        1D array containing frequency values. The numbers in the array f must be increasing and uniformly spaced
        (uniform frequency step). [Hz]
    sVarDens : NDArray[float64]
        1D array containing variance density spectrum of the signal [m^2/Hz]

    Returns
    -------
    float
        spectral period [s]

    """
    f, _ = core_engine.convert_to_vector(f)
    sVarDens, _ = core_engine.convert_to_vector(sVarDens)

    # --- calculate the spectral period (TPD) (s)
    max_spectum = max(sVarDens) * 0.8
    itemp = np.where(sVarDens / max_spectum >= 0.8)[0]
    temp = f[itemp]
    fp_limits = [min(temp), max(temp)]

    #  --- compute zeroth and first moment for selected frequency interval.
    m0 = compute_moment(f, sVarDens, 0, fp_limits[0], fp_limits[1])
    m1 = compute_moment(f, sVarDens, 1, fp_limits[0], fp_limits[1])

    # --- calculate TpD based on spectral moments.
    return m0 / m1


def compute_tps(f: NDArray[float64], sVarDens: NDArray[float64]) -> float:
    """Computes smoothed peak period.

    This function computes the smoothed peak period Tps, by means of quadratic interpolation, of a given variance
    density spectrum S = S(f).

    Parameters
    ----------
    f : NDArray[float64]
        1D array containing frequency values. The numbers in the array f must be increasing and uniformly spaced
        (uniform frequency step). [Hz]
    sVarDens : NDArray[float64]
        1D array containing variance density spectrum of the signal [m^2/Hz]

    Returns
    -------
    float
        smoothed peak period [s]

    Raises
    ------
    ValueError
        Input error: input should be 1d arrays
    ValueError
        Input error: frequency input parameter must be monotonic with constant step size
    ValueError
        Input error: array sizes differ in dimension

    Example
    -------
    >>> import numpy as np
    >>> f=np.arange(0,2,0.01)
    >>> Tp = 5.0
    >>> hm0 =1.0
    >>> sPM = create_spectrum_piersonmoskowitz(f,1/Tp,hm0)
    >>> Tps = compute_tps(f,sPM)

    """
    # --- Ensure array input is of type ndarray.
    f, fSize = core_engine.convert_to_vector(f)
    sVarDens, SSize = core_engine.convert_to_vector(sVarDens)

    if fSize[1] > 1 or SSize[1] > 1:
        raise ValueError("compute_moment: Input error: input should be 1d arrays")

    if fSize[1] > 1 and not core_engine.monotonic_increasing_constant_step(f):
        raise ValueError(
            "compute_moment: Input error: frequency input parameter must be monotonic with constant step size"
        )

    if not (fSize[0] == SSize[0]):
        raise ValueError("compute_moment: Input error: array sizes differ in dimension")

    Smax = max(sVarDens)
    if Smax < 1e-10:
        Tps = -999
        return Tps
    nF = fSize[0]
    imax = np.where(sVarDens == Smax)[0]
    imax = imax.astype(int)
    nmax = len(imax)

    # --- Depending on value of nF, compute Tps
    if nF > 2:
        # --- nF > 2 - default situation
        if nmax == 1:
            # --- nmax = 1
            jmax = imax
            if imax == 0:
                jmax = 1
            elif imax == nF - 1:
                jmax = nF - 2

            # --- Find polynomial coefficients
            ff = np.asarray([f[jmax - 1], f[jmax], f[jmax + 1]]).reshape(1, 3)[0]
            ee = np.asarray(
                [sVarDens[jmax - 1], sVarDens[jmax], sVarDens[jmax + 1]]
            ).reshape(1, 3)[0]
            p = np.polyfit(ff, ee, 2)
            a = p[0]
            b = p[1]
            # --- Compute Fps
            if a < 0.0:
                Fps = -b / (2 * a)
            else:
                # Exceptional situation; can only occur if imax=1 or imax=nF
                Fps = f[imax]
            Tps = 1.0 / Fps

        elif nmax == 2:
            # --- nmax = 2
            if (imax[1] - imax[0]) == 1:
                # Points are neighbours
                if imax[0] == 0:
                    kmax = 1
                elif imax[1] == nF - 1:
                    kmax = nF - 2
                else:
                    kmax = imax[0]
            else:
                # Points are not neighbours - make arbitrary choice
                Tps = 1 / f[imax[0]]
                return Tps

            # --- Find polynomial coefficients
            ff = np.asarray([f[kmax - 1], f[kmax], f[kmax + 1]]).reshape(1, 3)[0]
            ee = np.asarray(
                [sVarDens[kmax - 1], sVarDens[kmax], sVarDens[kmax + 1]]
            ).reshape(1, 3)[0]
            p = np.polyfit(ff, ee, 2)
            a = p[0]
            b = p[1]
            # --- Compute Fps (note: in this case, a < 0 always)
            Fps = -b / (2 * a)
            Tps = 1 / Fps
        else:
            # --- nmax >= 3 - make arbitrary choice
            Tps = 1 / f[imax[1]]
    elif nF == 2:
        # --- nF = 2: two points
        if nmax == 1:
            # nmax = 1
            Tps = 1 / f[imax]
        else:
            # nmax = 2
            favg = 0.5 * (f[0] + f[1])
            Tps = 1 / favg

    else:
        # --- nF = 1: one point
        Tps = 1 / f[0]

    return Tps


def compute_BattjesGroenendijk_wave_height_distribution(
    hm0: float,
    nwave: int,
    water_depth: float,
    cota_slope: float = 250.0,
    tolerance: float = 1e-5,
) -> tuple[NDArray[float64], NDArray[float64]]:
    """Computes wave height distribution following Battjes and
    Groenendijk (2000)

    Wave height distribution for sloping, shallow bottoms, as proposed by Battjes and Groenendijk (2000).

    Parameters
    ----------
    hm0 : float
        spectral wave height [m]
    nwave : int
        number of waves [-]
    water_depth : float
        water depth [m]
    cota_slope : float, optional
        cotangent of the bottom slope [-], by default 250.0
    tolerance : float, optional
        tolerance for convergence of transition wave height [-], by default 1e-5

    Returns
    -------
    tuple[NDArray[float64], NDArray[float64]]
        hWave_BG : NDArray[float64]
            theoretical Battjes & Groenendijk (2000) wave height distribution [m]
        Pexceedance_BG : NDArray[float64]
            theoretical Battjes & Groenendijk (2000) wave height exceedance probability [-]

    Example
    -------
    >>> Hm0 = 2.0
    >>> nwave = 700
    >>> water_depth = 4.5
    >>> cota_slope = 50
    >>> tolerance = 1e-5
    >>> hwave_BG, Pexceedance_BG = compute_BattjesGroenendijk_wave_height_distribution(Hm0, nwave, water_depth,
        cota_slope=cota_slope, tolerance=tolerance)

    """
    if cota_slope < 20 or cota_slope > 250:
        warnings.warn(
            UserWarning(
                "The foreshore slope used as input is not within the validity range (between 1:250 and 1:20) reported "
                "by Battjes & Groenendijk (2000)"
            ),
            stacklevel=2,
        )

    gamma_transition = 0.35 + 5.8 * (1 / cota_slope)
    H_transition = gamma_transition * water_depth
    m0 = np.power(hm0 / 4, 2)
    H_rms = np.sqrt(m0) * (2.69 + 3.24 * np.sqrt(m0) / water_depth)

    H_transition_norm = H_transition / H_rms

    if H_transition_norm > 2.75:
        warnings.warn(
            UserWarning(
                "The normalized transition wave height is larger than 2.75, which means the wave heights are Rayleigh "
                "distributed instead of the distribution reported by Battjes & Groenendijk (2000)"
            ),
            stacklevel=2,
        )

        hwave_BG = np.empty((0, 0))
        Pexceedance_BG = np.empty((0, 0))
    else:
        delta_H = 0.01
        H_1_norm = 100
        k1 = 2
        k2 = 3.6
        EST = 10
        while abs(EST - 1) > tolerance:
            H_2_norm = H_transition_norm / np.power(
                H_transition_norm / H_1_norm, k1 / k2
            )
            A_1 = 2 / k1 + 1
            X = np.power(H_transition_norm / H_1_norm, k1)
            A_2 = 2 / k2 + 1
            EST = np.power(H_1_norm, 2) * special.gammainc(A_1, X) * special.gamma(
                A_1
            ) + np.power(H_2_norm, 2) * (
                special.gamma(A_2) - special.gamma(A_2) * special.gammainc(A_2, X)
            )  # eq (7) (gamma(A_1)=1, since A_1=2)
            H_1_norm = H_1_norm - delta_H * (EST - 1)

        H_1_norm = H_1_norm + delta_H * (EST - 1)
        x_1 = H_1_norm * np.power(np.log(nwave), 1 / k1)
        x_2 = H_2_norm * np.power(np.log(nwave), 1 / k2)
        if x_1 < H_transition_norm:
            x = x_1
        else:
            x = x_2

        x = x * H_rms
        x = min(x, exponweib.ppf(1 - 1 / nwave, 1, 2, scale=np.sqrt(8)) / 4 * hm0)

        # dist = "B&G"
        H_1 = H_1_norm * H_rms
        H_2 = H_2_norm * H_rms

        if hm0 < H_transition:
            P_H_tr = np.exp(-np.power(H_transition / H_1, k1))
        else:
            P_H_tr = np.exp(-np.power(H_transition / H_2, k2))

        hwave_BG = np.array([0.0, H_transition, float(x)])
        Pexceedance_BG = np.array([1, float(P_H_tr), 1 / nwave])

    return hwave_BG, Pexceedance_BG
