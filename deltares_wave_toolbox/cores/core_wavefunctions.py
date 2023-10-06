# --- python modules
import scipy.integrate as integrate
import numpy as np


# --- toolbox modules
import deltares_wave_toolbox.cores.core_engine as engine_core
import deltares_wave_toolbox.spectrum as spectrum


def compute_spectrum_params(f=None, S=None, fmin=None, fmax=None):
    """
    COMPUTE_SPECTRUM_PARAMS  Computes spectral parameters of given spectrum

    This function computes several spectral wave parameters of a given 1D
    spectrum


    Parameters
    ----------
    f    : array double (1D)
         1D array representing frequency axis (unit: Hz)
    S    : array double (1D)
         1D array representing variance density spectrum (units: m2/Hz).
    fmin : double
         (optional argument) lower bound of the moment integral (unit: Hz)
    fmax : double
         (optional argument) upper bound of the moment integral (unit: Hz)

    Returns
    -------
    Hm0   : double
          wave height (units: m)
    Tp    : double
          peak period (units: s)
    Tps   : double
          smoothed peak period (units: s)
    Tmm10 : double
          wave period based on (-1) and (0) moments (units: s)
    Tm01  : double
          wave period based on (0) and (1) moments (units: s)
    Tm02  : double
          wave period based on (0) and (2) moments (units: s)


    Syntax:
          [Hm0,Tp,Tps,Tmm10,Tm01,Tm02] = compute_spectrum_params(f,S,fmin,fmax)


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

    See also computemoment

    """

    # --- Ensure array input is of type ndarray.
    f, fSize = engine_core.convert_to_vector(f)
    S, SSize = engine_core.convert_to_vector(S)

    if fSize[1] > 1 or SSize[1] > 1:
        raise ValueError(
            "compute_spectrum_params: Input error: input should be 1d arrays"
        )

    if not engine_core.monotonic_increasing_constant_step(f):
        raise ValueError(
            "compute_spectrum_params: Input error: frequency input parameter must be monotonic with constant step size"
        )

    if not (fSize[0] == SSize[0]):
        raise ValueError(
            "compute_spectrum_params: Input error: array sizes differ in dimension"
        )

    #
    # --- Find values of fmin and fmax
    if fmin is None:
        fmin = f[0]
    if fmax is None:
        fmax = f[fSize[0] - 1]

    # --- Compute moments
    m_1 = compute_moment(f, S, -1, fmin, fmax)
    m0 = compute_moment(f, S, 0, fmin, fmax)
    m1 = compute_moment(f, S, 1, fmin, fmax)
    m2 = compute_moment(f, S, 2, fmin, fmax)

    # --- Compute wave height -----------------------------------------------
    Hm0 = 4 * np.sqrt(m0)

    # --- Put values to -999 (exception value) in situation that wave height is
    #     (virtually) zero
    if Hm0 < 1e-6 or np.isnan(Hm0):
        Hm0 = np.nan
        Tp = np.nan
        Tps = np.nan
        Tmm10 = np.nan
        Tm01 = np.nan
        Tm02 = np.nan
        return [Hm0, Tp, Tps, Tmm10, Tm01, Tm02]

    # --- Compute mean wave periods -----------------------------------------
    Tmm10 = m_1 / m0
    Tm01 = m0 / m1
    Tm02 = np.sqrt(m0 / m2)

    # --- Make separate arrays containing only part corresponding to
    #     frequencies between fmin and fmax
    #     Note: first zero selects first element of tuple.
    iFmin = np.where(f >= fmin)[0][0]  # matlab find( f>= fmin,1,'first');
    iFmax = np.where(f <= fmax)[0][-1]  # matlab find( f<= fmax,1,'last');
    fMiMa = f[iFmin : iFmax + 1]
    SMiMa = S[iFmin : iFmax + 1]

    # --- Compute peak period -----------------------------------------------
    Smax = max(SMiMa)
    imax = np.where(SMiMa == Smax)[0]  # matlab find( SMiMa == Smax );
    imax = imax.astype(int)
    ifp = max(imax)
    fp = fMiMa[ifp]
    #
    if np.all(ifp is None):  # matlab isempty(ifp)
        ifp = 1
        fp = fMiMa[ifp]
    Tp = 1 / fp

    # --- Compute smoothed peak period --------------------------------------
    Tps = compute_tps(fMiMa, SMiMa)

    return [Hm0, Tp, Tps, Tmm10, Tm01, Tm02]


def compute_moment(f=None, S=None, m=None, fmin=None, fmax=None):
    """
    COMPUTE_MOMENT  Computes the spectral moment

    This function computes the m'th order spectral moment
    of variance density spectrum S=S(f), with f the frequency axis,
    over frequency domain [fmin,fmax].

    It is required that fmin >= f_in(1).

    It is not required to have fmax <= f_in(end). So it ok to have fmax =
    Inf.
    If fmax>f(end), then the moment consists of the summation of two parts:
    (1) Integration of (f_in^m * S), with given S, over [fmin,f(end)]
    (2) Exact integration of (f^m * S_lim) over [f(end),fmax], where S_lim
        is a high-frequency f^(-5) tail.
    Typically, in such cases one puts fmax = Inf.


    Parameters
    ----------
    f     : array double (1D)
          1D array representing frequency axis (unit: Hz)
    S     : array double (1D)
          1D array representing variance density spectrum (units: m2/Hz).
    m     : integer
          order of moment (integer value)
    fmin  : double
          (optional argument) lower bound of the moment integral (unit: Hz)
    fmax  : double
          (optional argument) upper bound of the moment integral (unit: Hz)

    Returns
    -------
    moment : double
           the computed moment

    Syntax:
       moment = compute_moment(f,S,m,fmin,fmax)

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

    See also wavelab function computemoment, integral1d
    """

    # --- Ensure array input is of type ndarray.
    f, fSize = engine_core.convert_to_vector(f)
    S, SSize = engine_core.convert_to_vector(S)

    if fSize[1] > 1 or SSize[1] > 1:
        raise ValueError("compute_moment: Input error: input should be 1d arrays")

    if not engine_core.monotonic_increasing_constant_step(f):
        raise ValueError(
            "compute_moment: Input error: frequency input parameter must be monotonic with constant step size"
        )

    if not (fSize[0] == SSize[0]):
        raise ValueError("compute_moment: Input error: array sizes differ in dimension")

    # --- Make sure that f and S are either both column vectors or both row vectors
    # if ( size(f_in,1) == size(S,1) ):
    #    f = f_in
    # else
    #    f = f_in'

    # --- Remove the possible situation that f=0 in combination with m<0. This
    #     would lead to division by zero
    if m < 0 and f[0] == 0:
        freq = f[1:]  # matlab  f(2:end);
        spec = S[1:]  # matlab  S(2:end);
    else:
        freq = f
        spec = S

    # --- Compute the integrand, that is the product of f^m * S  (using lambda is an alternatief for using def
    # <function name> :)
    func_integrand = (
        lambda freq, m, spec: freq ** (m) * spec
    )  # matlab freq.^(m) .* spec;
    integrand = func_integrand(freq, m, spec)

    # --- Depending on number of input arguments, compute the moment integral
    if fmin is None or fmax is None:  # integrate over all values in freq interval.
        moment = integrate.simps(integrand, freq)  # moment = integral1d(freq,integrand)
    else:  # integrate over all values in sub interval.
        # fmin and fmax are given
        fminn = fmin
        if m < 0 and f[0] == 0:  # matlab f(1)
            if fmin == 0:
                fminn = f[1]  # matlab f(2)

        #
        if (
            fmax <= freq[len(freq) - 1]
        ):  # matlab freq(end))      NOTE: USE len(freq) here instead of Nf because length is altered on lines 741
            ifminn = engine_core.approx_array_index(freq, fminn)
            ifmax = engine_core.approx_array_index(freq, fmax) + 1
            # due to range specification ifminn:ifmax. e.g. S[0:Nf] runs from S[0] to S[Nf-1] (S[Nf-1] has a value
            # and not S[Nf])
            moment = integrate.simps(integrand[ifminn:ifmax], freq[ifminn:ifmax])
            # moment = integral1d(freq,integrand,fminn,fmax);

        else:
            # 1: Integral over [fminn,freq(end)]
            ifminn = engine_core.approx_array_index(freq, fminn)
            ifmax = engine_core.approx_array_index(freq, freq[len(freq) - 1]) + 1
            # due to range specification ifminn:ifmax.  e.g. S[0:Nf] runs from S[0] to S[Nf-1] (S[Nf-1] has a value
            # and not S[Nf])
            moment1 = integrate.simps(
                integrand[ifminn:ifmax], freq[ifminn:ifmax]
            )  # moment1 = integral1d(freq,integrand,fminn,freq(end));
            # 2: Integral over [freq(end),fmax]
            #    Variance density spectrum in this range: C * f^power, with
            #    C determined by power and spec(end)
            power = -5  # Power of high-frequency tail
            C = spec[len(spec) - 1] / (
                freq[len(freq) - 1] ** power
            )  # matlab C       = spec(end) / ( freq(end)^power );
            moment2 = (C / (m + power + 1)) * (
                fmax ** (m + power + 1) - freq[len(freq) - 1] ** (m + power + 1)
            )  # matlab moment2 = (C / (m+power+1)) * (fmax^(m+power+1) - freq(end)^(m+power+1));
            # Add the two moments
            moment = moment1 + moment2

    #
    return moment


def create_spectrum_jonswap(
    f=None, fp=None, hm0=None, gammaPeak=3.3, l_fmax=0, output_object=True
):
    """
    CREATE_SPECTRUM_JONSWAP  Creates a Jonswap spectrum

    This function creates the Jonswap variance density spectrum, based on a
    given frequency axis, wave height, peak frequency and peak enhancement
    factor.
    Literature:
    Hasselman, K., e.a. (1973), Erga"nzungsheft zur Deutschen
    Hydrographischen Zeitschrift, Reihe A(8), No. 12


    Parameters
    ----------
    f : TYPE
        DESCRIPTION.
    fp : TYPE
        DESCRIPTION.
    hm0 : TYPE
        DESCRIPTION.
    gammaPeak : TYPE
        DESCRIPTION.
    l_fmax : TYPE
        DESCRIPTION.
    f         : array double (1D)
              1D real array containing frequency values. The numbers in
              the array f must be increasing and uniformly spaced
              (uniform frequency step). Units: Hz
    fp        : double
              peak frequency. Units: Hz
    hm0       : double
              wave height. Units: m
    gammaPeak : double
              (optional parameter) peak enhancement factor
              Default value is 3.3. No units.
    l_fmax    : double
              optional argument. The imposed spectral wave height Hm0
              holds for the frequency range [f(1),f(end)] (l_fmax = 0,
              default) or for the frequency range [f(1),inf] (l_fmax =
              1).

    Returns
    -------
    sVarDens = 1D array containing variance density (units m^2/Hz)
    None.

    For l_fmax = 0, the sVarDens is such that integration from f(1) to f(end)
    leads exactly to the given Hm0.
    For l_fmax = 1, integration from [f(end},inf] is computed using a
    (-5)-power law. This also means that integration from f(1) to f(end)
    leads to a slightly smaller value for the wave height than the
    prescribed Hm0.

    Syntax:
    sVarDens = createspectrumjonswap(f,fp,hm0,gammaPeak,l_fmax)

    Example
    -------
    >>> import numpy as np
    >>> f=np.arange(0,2,0.1)
    >>> Tp = 5.0
    >>> hm0 =1.5
    >>> S = create_spectrum_jonswap(f,1/Tp,hm0,3.3)

    See also create_spectrum_piersonmoskowitz, create_spectrum_tma,
    spectrum2_timeseries


    """

    # --- Ensure array input is of type ndarray.
    f, fSize = engine_core.convert_to_vector(f)

    nf = fSize[0]

    # Perform check on input arguments
    # --- Check whether input array f is a 1D vector array
    isvalid_size = nf > 0  # matlab ( numel(size(f))~=2 || min(size(f)) ~= 1 )
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
    for iff in np.arange(0, nf):  # matlab 1:length(f);
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

    if output_object:
        return spectrum.Spectrum(f, sVarDens)
    else:
        return sVarDens


def create_spectrum_piersonmoskowitz(
    f=None, fp=None, hm0=None, l_fmax=0, output_object=True
):
    """

    CREATE_SPECTRUM_PIERSONMOSKOWITZ  Creates a Pierson-Moskowitz spectrum

    This function creates the Pierson-Moskowitz variance density spectrum,
    based on agiven frequency axis, wave height and peak frequency. The
    Pierson-Moskowitz spectrum is identical to the Jonswap spectrum with a
    peak enhancement factor equal to 1. Furthermore, the Pierson-Moskowitz
    spectrum, the Bretschneider spectrum and the ITTC spectrum are all
    three identical.
    Literature:
    Pierson, W.J. and L. Moskowitz (1964). A proposed spectral form for
    fully developed wind seas based on the similarity theory of S.A.
    Kitaigorodskii. Journal of Geophysical Research,Vol. 69, No. 24, pg.
    5181 - 5190.

    Parameters
    ----------
    f         : array double (1D)
              1D real array containing frequency values. The numbers in
              the array f must be increasing and uniformly spaced
              (uniform frequency step). Units: Hz
    fp        : double
              peak frequency. Units: Hz
    hm0       : double
              wave height. Units: m
    l_fmax    :
              optional argument. The imposed spectral wave height Hm0
              holds for the frequency range [f(1),f(end)] (l_fmax = 0,
              default) or for the frequency range [f(1),inf] (l_fmax =
              1).


    Returns
    -------
    sVarDens : array double (1D)
             1D array containing variance density (units m^2/Hz)

    For l_fmax = 0, the sVarDens is such that integration from f(1) to f(end)
    leads exactly to the given Hm0.
    For l_fmax = 1, integration from [f(end},inf] is computed using a
    (-5)-power law. This also means that integration from f(1) to f(end)
    leads to a slightly smaller value for the wave height than the
    prescribed Hm0.


    Syntax:
    sVarDens = create_spectrum_piersonmoskowitz(f,fp,hm0,l_fmax)

    Example:
    >>> import numpy as np
    >>> f=np.arange(0,2,0.01)
    >>> Tp = 5.0
    >>> hm0 =1.5
    >>> Spm = create_spectrum_piersonmoskowitz(f,1/Tp,hm0)

    See also create_spectrum_jonswap, create_spectrum_tma,
    spectrum2_timeseries

    """

    # --- Ensure array input is of type ndarray.
    f, fSize = engine_core.convert_to_vector(f)

    # Computational core
    # --- Use the fact that the Pierson-Moskowitz spectrum is identical to the
    #     Jonswap spectrum with a peak enhancement factor equal to 1.
    gammaPeak = 1
    sVarDens = create_spectrum_jonswap(f, fp, hm0, gammaPeak, l_fmax)

    if output_object:
        return spectrum.Spectrum(f, sVarDens)
    else:
        return sVarDens


def tpd(freqs: np.ndarray = None, spectrum: np.ndarray = None) -> float:
    """
    TpD : Function which calculates the spectral period (s)

    Input (key,value) :
    'Spectrum',value   : numeric, array of variance density values (m2/Hz)
    'Frequencies',value: numeric, array of frequencies (Hz)
     Output :
     out                : numeric, spectral period (s)

    Note: For definition of TpD: Overstap van piekperiode naar spectrale periode bij ontwerp van steenzettingen
    """

    freqs, f_size = engine_core.convert_to_vector(freqs)
    spectrum, spectrum_size = engine_core.convert_to_vector(spectrum)

    # --- calculate the spectral period (TPD) (s)
    max_spectum = max(spectrum) * 0.8
    itemp = np.where(spectrum / max_spectum >= 0.8)[
        0
    ]  # temp=freqs( (spectrum ./ maxSpectum) >= 0.8);
    temp = freqs[itemp]
    fp_limits = [min(temp), max(temp)]  # clear temp

    #  --- compute zeroth and first moment for selected frequency interval.
    m0 = compute_moment(freqs, spectrum, 0, fp_limits[0], fp_limits[1])
    m1 = compute_moment(freqs, spectrum, 1, fp_limits[0], fp_limits[1])

    # --- calculate TpD based on spectral moments.
    return m0 / m1


def compute_tps(f=None, S=None) -> float:
    """
    COMPUTE_TPS  Computes smoothed peak period.

    This function computes the smoothed peak period Tps, by means of
    quadratic interpolation, of a given variance density spectrum S = S(f).


    Parameters
    ----------
    f    : array double (1D)
         1D array representing frequency axis (unit: Hz)
    S    : array double (1D)
         1D array representing variance density spectrum (units: m2/Hz).


    Returns
    -------
    Tps  : double
         smoothed peak period (units: s)

    Syntax:
    Tps = compute_tps(f,S)


    Example
    >>> import numpy as np
    >>> f=np.arange(0,2,0.01)
    >>> Tp = 5.0
    >>> hm0 =1.0
    >>> sPM = create_spectrum_piersonmoskowitz(f,1/Tp,hm0)
    >>> Tps = compute_tps(f,sPM)


    See also compute_spectrum_params

    """

    # --- Ensure array input is of type ndarray.
    f, fSize = engine_core.convert_to_vector(f)
    S, SSize = engine_core.convert_to_vector(S)

    if fSize[1] > 1 or SSize[1] > 1:
        raise ValueError("compute_moment: Input error: input should be 1d arrays")

    if fSize[1] > 1 and not engine_core.monotonic_increasing_constant_step(f):
        raise ValueError(
            "compute_moment: Input error: frequency input parameter must be monotonic with constant step size"
        )

    if not (fSize[0] == SSize[0]):
        raise ValueError("compute_moment: Input error: array sizes differ in dimension")

    Smax = max(S)
    if Smax < 1e-10:
        Tps = -999
        return Tps
    nF = fSize[0]
    # note: [0] selects first part of the tuple.
    imax = np.where(S == Smax)[0]  # matlab find( S == Smax );
    imax = imax.astype(int)
    nmax = len(imax)

    # --- Depending on value of nF, compute Tps
    if nF > 2:
        # --- nF > 2 - default situation
        if nmax == 1:
            # --- nmax = 1
            # matlab imax   ->in matlab jmax=imax, as imax=np.where() it already account for the fact that indices
            # starts with zero.
            jmax = imax
            if imax == 0:
                jmax = 1  # the one after the first one: starting at 0, so jmax must be 1 in python ->matlab jmax=2
            elif imax == nF - 1:
                jmax = nF - 2  # the one before the last one: matlab nF-1

            # --- Find polynomial coefficients. note: due to double brackets reduce dimension by selecting [0]
            # ff = np.array([f[jmax-1],f[jmax],f[jmax+1]]).reshape(1,3)[0]
            ff = np.asarray([f[jmax - 1], f[jmax], f[jmax + 1]]).reshape(1, 3)[0]
            ee = np.asarray([S[jmax - 1], S[jmax], S[jmax + 1]]).reshape(1, 3)[0]
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

            # --- Find polynomial coefficients. note: due to double brackets reduce dimension by selecting [0]
            ff = np.asarray([f[kmax - 1], f[kmax], f[kmax + 1]]).reshape(1, 3)[0]
            ee = np.asarray([S[kmax - 1], S[kmax], S[kmax + 1]]).reshape(1, 3)[0]
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

    return Tps  # TODO: check for some reason single value is stored as array,
