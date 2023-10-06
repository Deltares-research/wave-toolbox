# --- python modules
import numpy as np
import numbers
import re
import sys
from scipy.interpolate import interp1d


# --- functions
# ---- see https://stackoverflow.com/questions/736043/checking-if-a-string-can-be-converted-to-float-in-python
def is_float(element: any) -> bool:
    # If you expect None to be passed:
    if element is None:
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False


def str2num(string):
    return np.asfarray(string, dtype=float)


def contains(src, substr):
    if substr in src:
        return True
    return False


def is_of_type_float(**kwargs):
    for key, value in kwargs.items():
        if not (isinstance(value, float)):
            raise ValueError("Parameter" + key + " must be of type float")


def genvarname(h):
    # Remove invalid characters
    if not iscell(h):
        h = {h}

    par = set()
    for item in h:
        s = re.sub(r"[^0-9a-zA-Z_]", "", item)
        # Remove leading characters until we find a letter or underscore
        s = re.sub(r"^[^a-zA-Z_]+", "", s)
        s = re.sub(r"\W+|^(?=\d)", "_", s)
        par.add(s.lower())

    return par


def convert_to_array_type(x):
    if isinstance(x, numbers.Number):
        x = np.asarray(
            [x]
        )  # for some reason in this case the statement x = np.asarray(x) is not sufficient.
    elif not isinstance(x, (np.ndarray)):
        x = np.asarray(x)
    return x


def convert_to_vector(x):
    if isinstance(x, numbers.Number):
        x = np.asarray(
            [x]
        )  # for some reason in this case the statement x = np.asarray(x) is not sufficient.
    elif not isinstance(x, (np.ndarray)):
        x = np.asarray(x)

    # ensure vector convention if 1d array is used either (n,1) or (n,)
    xSize = _size(x)
    if xSize[1] == 1:
        # ensure 1d vector format of (n,) instead of (n,1)
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


def _len(x):
    lenx = 0
    if x is not None:
        if isinstance(x, list):
            lenx = len(x)
        elif isinstance(x, np.ndarray):
            # todo: need to handle lenx =0 and leny!=0 and lenx >0 and leny>0
            lenx, leny = _size(x)
    return lenx


def ischar(h):
    return h.isalpha()


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


def iscell(h):
    return isinstance(h, set)


def is1darray(x):
    is1d = False
    dimx, dimy = _size(x)
    if dimx != 0 and dimy == 0:
        is1d = True
    return is1d


def approx_array_index(array, user_value):
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


def is_digit(inStr):
    """
    Returns a list of True/False on the location of a digit (or .) (True) in a string

    Parameters
    ----------
    inStr : str
        String to identify digits in.


    Returns
    -------
    outIndx : list
        True/False at locations of Digits/Characters.

    """

    if not isinstance(inStr, str):
        raise TypeError("Input should be of type str")

    outIndx = list()
    for indx, char in enumerate(inStr):
        if char.isdigit() or char == ".":
            outIndx.append(True)
        else:
            outIndx.append(False)
    return outIndx


def list_of_strings_upper(in_list_strings):
    """
    Returns the input list of strings (only strings) all upper case

    Parameters
    ----------
    in_list_strings : TYPE
        DESCRIPTION.

    Returns
    -------
    out_list_strings : TYPE
        DESCRIPTION.

    """

    out_list_strings = list()
    for el in in_list_strings:
        out_list_strings.append(el.upper())

    return out_list_strings


def rfwave(h=None, T=None, H=None):
    """
    rfwave non-linear Rienecker-Fenton wave estimates


    -------------------------------------------------------------
    Copyright (c) Deltares 2010 FOR INTERNAL USE ONLY
    Version:      Beta Version 2.00, <2012 April 13>
    -------------------------------------------------------------


    Parameters
    ----------
    h        :
              water depth (m)
    T        :
             wave period
    H        :
             wave height

    Returns
    -------
    eta_max :
            maximal crest height according to Rienecker-Fenton
    ua      :
            orbital velocity at seabed according to Rienecker-Fenton
    ua_lin  :
            orbital velocity at seabed according to linear wave theory
    L       :
            wave length according to Rienecker-Fenton
    L_lin   :
            wave length according to linear wave theory

    Syntax:   [eta_max, ua, ua_lin, L, L_lin] = rfwave(h, T, H)  ;

    See also oBGinv.m


    0. Authors:
        11/2007: Peter Wellens, creation
        1/2009: SC, introduction of help
    1. Method:
       Rienecker-Fenton (1981)

    """

    accuracy = 1e-4
    t = 0  # time
    xeta = 0  # location along x-axis where sea level elevation is needed
    xvel = 0  # location along x-axis where velocity is needed
    zvel = 0  # location along z-axis where velocity is needed
    g = 9.81  # gravitational acceleration

    # now first the input will be checked, based on the combination of h, T and
    # H. Rfwave cannot handle waves that are too large. Because Rfwave is a 'blackbox'
    # the wrong combinations of h, H and T will be checked on beforehand.
    # The only available information is a certain dimensionless line of 'limit values', which is described below.

    # xLimit is T*sqrt(g/h) and yLimit is H/h
    # the table of Klopman stops at [25.6; 0.786], why is still unclear, but
    # therefore the yLimit-values have set to 0 for larger xLimits.
    # More research is necessary!!
    xLimit = [
        50,
        25.61,
        25.6,
        20.8,
        17.6,
        15.3,
        13.5,
        12.2,
        11.1,
        9.53,
        8.39,
        7.55,
        6.89,
        6.37,
        5.59,
        5.05,
        4.64,
        4.33,
        4.08,
        3.62,
        0,
    ]
    yLimit = [
        0,
        0,
        0.786,
        0.775,
        0.765,
        0.756,
        0.747,
        0.737,
        0.728,
        0.709,
        0.689,
        0.668,
        0.646,
        0.623,
        0.576,
        0.53,
        0.487,
        0.447,
        0.412,
        0.342,
        0,
    ]

    # the combination of input values is used to define xT and yH.
    xT = T * np.sqrt(g / h)
    yH = H / h
    p = [xT, yH]

    # The function "dist2func" is used to check whether the dimensionless point (xT,yH) is situated below or
    # above this limit-line. For extra safety (see Klopman, 1989) a safety-factor is introduced:
    safP = 0.95
    d = dist2func(xLimit, safP * yLimit, p)

    # if d < 0 point p is situated below the limit-line and rfwave can be used.
    if d < 0:
        # disp('Rfwave can and will be used')
        # make inputfile for rfwave
        [eta, u, w, L] = rfwave_rfw(T, t, H, h, xeta, xvel, zvel)
        eta_max = eta[0, 0, 0] - h
        ua = u[0, 0, 0]
        WT = 0
        # if d>=0 rfwave cannot be used and SPP will use linear wave theory
    else:
        print(
            "The wave characteristics fall outside the range of Rfwave. Only the results of linear wave theory are shown. Be cautious with the results! Is your input correct?"
        )
        # warndlg('The wave characteristics fall outside the range or Rfwave. Now linear wave theory will be used. Be cautious with the results! Is your input correct?')
        # use linear wave theory, therefore make input for spp_calc_l.m to
        # compute wave length
        L = 0
        ua = 0
        eta_max = 0
        # this variable tells which wave theory has been used. "1" means that
        # linear theory has been used.
        WT = 1

    # calculate values from linear wave theory
    L_lin = calc_l_lin(T, h, g)
    ua_lin = np.pi * H / (T * np.sinh(2 * np.pi * h / L_lin))

    # show results
    ua_rfwave = ua
    ua_linear = ua_lin
    L_rfwave = L
    L_linear = L_lin

    return eta_max, ua, ua_lin, L, L_lin


def dist2func(x=None, y=None, p=None):
    """
    DIST2FUNC: this function checks whether a certain point is situated above or below a
    certain line. This line is linearized between the two nearest points.


    if d > 0 then point p is situated above the line
    if d < 0 then point p is situated below the line

    Parameters
    ----------
    x       :
            array of x values of a certain line
    y       :
            array of y-values of a certain line
    p       :
            2-dimensional array (x,y) of a certain point

    Returns
    -------
    d      :
           distance to point.

    """
    xd = abs(x - p[0])
    id = np.argsort(xd)  # matlab [dummy, id]=sort(xd);

    # define function between two nearest points
    a = (y[id[1]] - y[id[0]]) / (x[id[1]] - x[id[0]])
    b = y[id[0]] - a * x[id[0]]
    py = a * p[0] + b

    d = p[1] - py

    # if d>0:
    #       # fprint('So point is situated above line');
    # elif d<0:
    #       # fprint('So point is situated below line');
    # elif d==0:
    #       # fprint('So point is situated exactly at the line');
    # else:
    #       # fprint('You silly bastard, your input was not correct!');

    return d


def rfwave_rfw(T=None, t=None, H=None, h=None, xeta=None, xvel=None, zvel=None):
    """
    RFWAVE - gives the characteristics of a rienecker-fenton wave. The
             function is used by giving the following command:
    (ORCA: )

    Parameters
    ----------
    T       :
            period                                                  [m/s]
    t       :
            time                                                    [m/s]
    H       :
            wave height                                               [m]
    h       :
            water depth                                               [m]
    xeta    :
            positions where surface elevation is to be determined. Input
            as column vector.                                         [m]
    xvel    :
            x of the positions where velocities are required. Input as
            column vector.                                            [m]
    zvel    :
            z of the positions where velocities are required. Input as
            row vector.                                               [m]

    Note: the H/h ratio in this code can be no higher that 0.35. Higher
    ratios can be obtained by using an RF solution as initial estimate
    instead of a linear wave. See RFCOEF.


    Returns
    -------

    Description of the output parameters:
    eta     :
            surface elevation matrix; x in rows, t in columns         [m]
    u       :
            velocity matrix; x in rows, z in columns, t in 3rd dim  [m/s]
    w       :
            velocity matrix; x in rows, z in columns, t in 3rd dim  [m/s]%
    Syntax:
          [eta, u, w] = spp_calc_rfwave_rfw(T, t, H, h, xeta, xvel, zvel);

    """

    # Constants. N is the number of Fourier coefficients, g is the gravity.
    N = 16
    g = 9.81

    # Determine the coefficients. Note that the input parameters are made
    # non-dimensional.
    [x, ct, ceta, B0, B, c, k, R] = rfwave_rfcoef(H / h, T * (g / h) ** (1 / 2), N)

    # Determine the surface elevation and the velocities at the given
    # locations as a function of time.
    eta = np.zeros((len(xeta), len(t)), dtype=float)
    u = np.zeros((len(xvel), len(zvel), len(t)), dtype=float)
    w = np.zeros((len(xvel), len(zvel), len(t)), dtype=float)
    for tt in np.arange(0, len(t)):  # matlab tt = 1:length(t)
        eta[:, tt] = rfwave_rfeta(
            N, h, ceta, k, xeta / h, c, t[tt] * (g / h) ** (1 / 2)
        )
        [u[:, :, tt], w[:, :, tt]] = rfwave_rfvel(
            B0, B, c, k, xvel / h, zvel / h, t[tt] * (g / h) ** (1 / 2)
        )

    # Give dimensions
    eta = eta * h
    u = u * (g * h) ** (1 / 2)
    w = w * (g * h) ** (1 / 2)
    k = k / h
    L = 2 * np.pi / k

    return eta, u, w, L


def rfwave_rfcoef(H=None, tau=None, N=None):
    """
    RFCOEF determines the surface elevation and the coefficients to determine
    the velocities in a non-linear Rienecker-Fenton wave. Only paremeters for
    half a period are calculated.

    Note that the dimensionless H can be no higher than 0.35 when converging
    from a linear solution. For higher values of H the initial solution is
    an already converged RF solution.

    Parameters
    ----------
    H : TYPE, optional
        DESCRIPTION. The default is None.
    tau : TYPE, optional
        DESCRIPTION. The default is None.
    N : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """

    # Constants
    cE = 0
    D = 1
    c = 1

    # Initialize the solution process
    numapprox = 0
    if H > min((0.05 * tau), 0.35):
        Hin = H
        H = min((0.05 * tau), 0.35)
        dH = Hin - H
    else:
        Hin = H
        dH = 1

    while H <= Hin:
        if numapprox == 0:
            #     Initial linear approximation when H<=0.30
            jj = np.arange(0, N)  # matlab 1:N;
            mm = np.arange(0, N).conj().T  # matlab [0:N]';
            eta = 1 + 0.5 * H * np.cos(mm * np.pi / N)
            differ = 100
            k = 2 * np.pi / (tau * c)
            while differ > abs(1e-10):
                c = np.sqrt(np.tanh(k) / k)
                kn = 2 * np.pi / (tau * c)
                differ = kn - k
                k = kn

            B0 = -c
            B = np.zeros((1, N), dtype=float)
            B[0] = -0.25 * H / (c * k)
            R = 1 + 0.5 * c**2
            Q = c

        z = 100
        numiter = 0

        while max(z) > 1e-12 and numiter < 100:
            #     Fill the f vector
            matLeft = np.ones((N + 1, 1)) * B
            matMid = np.sinh(eta * jj * k) / np.cosh(np.ones((N + 1, 1)) * jj * k * D)
            matRight = np.cos(mm * jj * np.pi / N)
            toSum = matLeft * matMid * matRight
            f = B0 * eta + sum(toSum, 2) + Q

            matLeft = np.ones((N + 1, 1)) * (B * jj)
            matMid = np.cosh(eta * jj * k) / np.cosh(np.ones((N + 1, 1)) * jj * k * D)
            matRight = np.cos(mm * jj * np.pi / N)
            toSum = matLeft * matMid * matRight
            u = B0 + k * sum(toSum, 2)

            matMid = np.sinh(eta * jj * k) / np.cosh(np.ones((N + 1, 1)) * jj * k * D)
            matRight = np.sin(mm * jj * np.pi / N)
            toSum = matLeft * matMid * matRight
            v = k * sum(toSum, 2)

            f = np.concatenate(
                (f, 0.5 * u**2 + 0.5 * v**2 + eta - R), axis=None
            )  #  [f; 0.5*u.^2 + 0.5*v.^2 + eta - R]
            f = np.concatenate(
                (f, 1 / (2 * N) * (eta[0] + eta[N] + 2 * sum(eta[1 : N + 1])) - 1),
                axis=None,
            )  #  [f; 1/(2*N)*(eta(1) + eta(N+1) + 2*sum(eta(2:N))) - 1];
            f = np.concatenate(
                (f, eta[0] - eta[N] - H), axis=None
            )  #  [f; eta(1) - eta(N+1) - H];
            f = np.concatenate(
                (f, k * c * tau - 2 * np.pi), axis=None
            )  #  [f; k*c*tau - 2*pi];
            f = np.concatenate((f, c - cE + B0), axis=None)  # [f; c - cE + B0];

            #     Fill the A matrix. For i = 1:N+1:
            dfideta = np.diag(u)
            dfidB0 = eta  # in the paper this derivative is -eta
            S1 = (
                np.sinh(eta * jj * k)
                / np.cosh(np.ones((N + 1, 1)) * jj * k * D)
                * np.cos(mm * jj * np.pi / N)
            )
            dfidBj = S1
            dfidc = np.zeros((N + 1, 1))
            dfidk = eta * (u - B0) / k - D * sum(
                np.ones((N + 1, 1)) * (jj * B * np.tanh(jj * k * D)) * S1, 2
            )
            dfidQ = np.ones((N + 1, 1))
            dfidR = np.zeros((N + 1, 1))
            A = np.concatenate(
                (dfideta, dfidB0, dfidBj, dfidc, dfidk, dfidQ, dfidR), axis=None
            )  # [dfideta dfidB0 dfidBj dfidc dfidk dfidQ dfidR];

            #     For i = N+2:2N+2
            C1 = (
                np.cosh(eta * jj * k)
                / np.cosh(np.ones((N + 1, 1)) * jj * k * D)
                * np.sin(mm * jj * np.pi / N)
            )
            C2 = (
                np.cosh(eta * jj * k)
                / np.cosh(np.ones((N + 1, 1)) * jj * k * D)
                * np.cos(mm * jj * np.pi / N)
            )
            S2 = (
                np.sinh(eta * jj * k)
                / np.cosh(np.ones((N + 1, 1)) * jj * k * D)
                * np.sin(mm * jj * np.pi / N)
            )
            dfideta = np.diag(
                1
                + u * k**2 * sum(np.ones((N + 1, 1)) * (jj**2 * B) * S1, 2)
                + v * k**2 * sum(np.ones((N + 1, 1)) * (jj**2 * B) * C1, 2)
            )
            dfidB0 = u  # in the paper this is -u
            dfidBj = u * jj * k * C2 + v * jj * k * S2
            dfidc = np.zeros((N + 1, 1))

            # dfidk   = u.*((u-B0)/k + k*eta.*sum(ones(N+1,1)*(jj.^2.*B).*S1,2) - ...
            #              k*D*sum(ones(N+1,1)*(jj.^2.*B.*tanh(jj*k*D)).*C2,2)) +    ...
            #              v.*(v/k + k*eta.*sum(ones(N+1,1)*(jj.^2.*B).*C1,2) -      ...
            #              k*D*sum(ones(N+1,1)*(jj.^2.*B.*tanh(jj*k*D)).*S2,2)
            #             );

            dfidk = u * (
                (u - B0) / k
                + k * eta * sum(np.ones((N + 1, 1)) * (jj**2 * B) * S1, 2)
                - k
                * D
                * sum(np.ones((N + 1, 1)) * (jj**2 * B * np.tanh(jj * k * D)) * C2, 2)
            ) + v * (
                v / k
                + k * eta * sum(np.ones((N + 1, 1)) * (jj**2 * B) * C1, 2)
                - k
                * D
                * sum(np.ones((N + 1, 1)) * (jj**2 * B * np.tanh(jj * k * D)) * S2, 2)
            )

            dfidQ = np.zeros((N + 1, 1), dtype=float)
            dfidR = -np.ones((N + 1, 1))
            A = np.concatenate(
                (A, dfideta, dfidB0, dfidBj, dfidc, dfidk, dfidQ, dfidR), axis=None
            )  # [A; dfideta dfidB0 dfidBj dfidc dfidk dfidQ dfidR];
            #     For i = 2N+3
            dfideta = np.concatenate(
                (1 / (2 * N), np.ones((1, N - 1)) / N, 1 / (2 * N)), axis=None
            )  # [1/(2*N) ones(1,N-1)/N 1/(2*N)];
            A = np.concatenate(
                (A, dfideta, np.zeros((1, N + 5))), axis=None
            )  # [A; dfideta zeros(1, N+5)];
            #     For i = 2N+4
            dfideta = np.zeros((1, N + 1))
            dfideta[0] = 1
            dfideta[N] = -1
            A = np.concatenate(
                (A, dfideta, np.zeros((1, N + 5))), axis=None
            )  # [A; dfideta zeros(1, N+5)];
            #     For i = 2N+5
            dfidc = k * tau
            dfidk = c * tau
            A = np.concatenate(
                (A, np.zeros((1, 2 * N + 2)), dfidc, dfidk, 0, 0), axis=None
            )  # [A; zeros(1,2*N+2) dfidc dfidk 0 0];
            #     For i = 2N+6
            dfidc = 1
            dfidB0 = 1
            A = np.concatenate(
                (A, np.zeros((1, N + 1)), dfidB0, np.zeros((1, N)), dfidc, 0, 0, 0),
                axis=None,
            )  # [A; zeros(1,N+1) dfidB0 zeros(1,N) dfidc 0 0 0];

            #     Solution
            z = np.linalg.solve(A, -f)  # z       = A\-f

            eta = z[0:N] + eta  # z(1:N+1) + eta;
            B0 = z[N + 1] + B0  # z(N+2) + B0;
            B = z[N + 2 : 2 * N + 2].T + B  # z(N+3:2*N+2)' + B;
            c = z[2 * N + 3 - 1] + c  # z(2*N+3) + c;
            k = z[2 * N + 4 - 1] + k  # z(2*N+4) + k;
            Q = z[2 * N + 5 - 1] + Q  # z(2*N+5) + Q;
            R = z[2 * N + 6 - 1] + R  # z(2*N+6) + R;

            #     Start new iteration
            numiter = numiter + 1

        H = H + dH
        numapprox = numapprox + 1
    x = np.pi * mm / (N * k)
    t = np.pi * mm / (N * c * k)

    return x, t, eta, B0, B, c, k, R


def rfwave_rfeta(N=None, h=None, ceta=None, k=None, x=None, c=None, t=None):
    """

    Perform DFT on the half period RF wave generated by rfcoef. This is done
    by making eta (=ceta) non-dimensional with k instead of h and subtracting
    the depth (which is 1 when made non-dimensional with h). Later eta is
    made non-dimensional again by h instead of k by dividing by kh and adding
    one!

    Parameters
    ----------
    N : TYPE, optional
        DESCRIPTION. The default is None.
    h : TYPE, optional
        DESCRIPTION. The default is None.
    ceta : TYPE, optional
        DESCRIPTION. The default is None.
    k : TYPE, optional
        DESCRIPTION. The default is None.
    x : TYPE, optional
        DESCRIPTION. The default is None.
    c : TYPE, optional
        DESCRIPTION. The default is None.
    t : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """

    x0 = 0.0
    jj = np.arange(0, N)  # 1:N;
    mm = np.arange[0:N].T  # [0:N]';
    ceta = k * (ceta - 1) * h
    Nceta = len(ceta)
    # TODO check index value -3
    Y = (
        2
        / N
        * (
            0.5 * ceta[0]
            + sum(
                ceta[0 : Nceta - 3]
                * np.ones((1, N))
                * np.cos(mm[0 : Nceta - 3] * jj * np.pi / N),
                1,
            )
            + 0.5 * ceta[-1] * np.cos(jj * np.pi)
        )
    )
    eta = (
        sum(
            np.ones((len(x), 1))
            * Y[0 : Nceta - 3]
            * np.cos((x - c * t) * jj[0 : Nceta - 3] * k),
            2,
        )
        + 0.5 * Y[-1] * np.cos(jj[-1] * k * (x - c * t))
    ) / (k * h) + 1

    return eta


def rfwave_rfvel(B0=None, B=None, c=None, k=None, x=None, z=None, t=None):
    """


    Parameters
    ----------
    B0 : TYPE
        DESCRIPTION.
    B : TYPE
        DESCRIPTION.
    c : TYPE
        DESCRIPTION.
    k : TYPE
        DESCRIPTION.
    x : TYPE
        DESCRIPTION.
    z : TYPE
        DESCRIPTION.
    t : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # Constants. D is the non-dimensional depth h/h.
    D = 1
    x0 = 0

    # Make the arrays 3D.
    JJ = []  # python required otherwise JJ unknown
    JJ[0, 0, :] = np.arange(0, len(B))  # JJ(1,1,:) = 1:length(B);
    JJ = np.tile(JJ, (len(x), len(z), 1))  # repmat(JJ,[length(x), length(z),1]);
    BB = []
    BB[0, 0, :] = B  # BB(1,1,:) = B;
    BB = np.tile(BB, (len(x), len(z), 1))  # repmat(BB,[length(x), length(z),1]);
    X = np.tile(x, (1, len(z), len(B)))  # repmat(x, [1, length(z), length(B)]);
    Z = np.tile(z, (len(x), 1, len(B)))  # repmat(z, [length(x), 1, length(B)]);

    # Determine the horizontal velocity u.
    UJ = (
        JJ
        * BB
        * np.cosh(Z * JJ * k)
        / np.cosh(JJ * k * D)
        * np.cos((X - x0 - c * t) * JJ * k)
    )  # JJ.*BB.*cosh(Z.*JJ*k)./cosh(JJ*k*D).*cos((X-x0-c*t).*JJ*k);
    u = c + B0 + k * sum(UJ, 3)
    UJ = []  # clear UJ;

    # Determine the vertical velocity v.
    WJ = (
        JJ
        * BB
        * np.sinh(Z * JJ * k)
        / np.cosh(JJ * k * D)
        * np.sin((X - x0 - c * t) * JJ * k)
    )  # JJ.*BB.*sinh(Z.*JJ*k)./cosh(JJ*k*D).*sin((X-x0-c*t).*JJ*k);
    w = k * sum(WJ, 3)

    return u, w


def calc_l_lin(T, h, g):
    """
    L = calc_l_lin(T,h,g)
    calculate wavelength given the period of the wave and the water depth
    using linear theory
    rewrite dispersion relation:
    y = x*tanh(x)
    y = omega^2*h/g
    x = k*h
    y is given
    f(x) = x*tanh(x) - y = 0
    f'(x) = tanh(x) + x*cosh(x)^-2
    now we can Newton-Raphson iteration:
    x(n+1) = x_n - f(x_n)/f'(x_n)

    Parameters
    ----------
    T : TYPE
        DESCRIPTION.
    h : TYPE
        DESCRIPTION.
    g : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    y = (2 * np.pi) ** 2 / T**2 * h / g  # omega^2*h/g
    x_o = 0.1  # old estimate
    x_n = 1  # new estimate
    #
    # iteration
    while abs((x_n - x_o)) / x_n > 1e-12:
        x_o = x_n
        fx = x_o * np.tanh(x_o) - y
        fpx = np.tanh(x_o) + x_o * np.cosh(x_o) ** -2
        x_n = x_o - fx / fpx

    L_lin = 2 * np.pi * h / x_n

    return L_lin


def str2num_if_numeric(value: str = None) -> [(int or float or complex), bool]:
    if (
        not isinstance(value, str)
        or isinstance(value, float)
        or isinstance(value, int)
        or isinstance(value, complex)
    ):
        number = value
        is_converted = False
    elif (
        isinstance(value, float) or isinstance(value, int) or isinstance(value, complex)
    ):
        number = value
        is_converted = True
    elif value.strip().isnumeric():  # basically detects integers, without "."
        number = int(value)
        is_converted = True
    elif value.strip() == "":
        number = None
        is_converted = True
    else:
        try:
            number = float(value)
            is_converted = True
        except:
            number = value
            is_converted = False

    return number, is_converted


def acceleration(t: np.ndarray = None, x: np.ndarray = None) -> np.ndarray:
    """
    ACCELERATION  Computes the acceleration from a time signal x = x(t)

    Syntax:
    a = acceleration(t,x)

    Input:
    t     = 1D real array containing time values. The numbers in the
            array t must be increasing and uniformly spaced (uniform time
            step). The initial time t(1) can be any value (so it is not
            obligatory to have t(1) = 0)
    x     = 1D real array containing signal values, i.e. the time
            series of the signal. The value xTime(i) must be the signal
            value at time t(i). The number of elements in t and xTime must
            be the same

    Output:
    a     = 1D real array containing values of the acceleration

    Example

    See also
    """

    t, t_size = convert_to_vector(t)
    x, x_size = convert_to_vector(x)

    # Computational core
    a = np.zeros(x_size[0], dtype=float)
    nTime = t_size[0]
    for i in np.arange(1, nTime - 1):  # matlab for i = 2:(nTime-1)
        t_p05 = 0.5 * (t[i] + t[i + 1])
        t_m05 = 0.5 * (t[i] + t[i - 1])
        v_p05 = (x[i + 1] - x[i]) / (t[i + 1] - t[i])
        v_m05 = (x[i] - x[i - 1]) / (t[i] - t[i - 1])
        #
        a[i] = (v_p05 - v_m05) / (t_p05 - t_m05)
    # Constant extrapolation
    a[0] = a[1]
    a[nTime - 1] = a[nTime - 2]
    return a


def velocity(t: np.ndarray = None, x: np.ndarray = None) -> np.ndarray:
    """
    VELOCITY  Computes the velocity from a time signal x = x(t)

    Syntax:
    v = velocity(t,x)

    Input:
    t     = 1D real array containing time values. The numbers in the
            array t must be increasing, but not necessarily uniformly spaced
            (nonuniform time step is allowed). The initial time t(1) can be
            any value (so it is not obligatory to have t(1) = 0)
    x     = 1D real array containing signal values, i.e. the time
            series of the signal. The value xTime(i) must be the signal
            value at time t(i). The number of elements in t and xTime must
            be the same

    Output:
    v     = 1D real array containing values of the velocity

    Example

    See also
    """

    t, t_size = convert_to_vector(t)
    x, x_size = convert_to_vector(x)

    # Computational core
    v = np.zeros(x_size[0], dtype=float)
    nTime = t_size[0]

    # Central elements
    for i in np.arange(1, nTime - 1):  # matlab for i = 2:(nTime-1)
        v[i] = (x[i + 1] - x[i - 1]) / (t[i + 1] - t[i - 1])

    # One-sided differences
    v[0] = (x[1] - x[0]) / (t[1] - t[0])
    v[nTime - 1] = (x[nTime - 1] - x[nTime - 2]) / (t[nTime - 1] - t[nTime - 2])
    return v


def interp_measurement_timeseries(
    in_time: list or np.ndarray,
    in_signal: list or np.ndarray,
    new_time: list or np.ndarray,
):
    """
    Interpolates one timeseries onto another

    Parameters
    ----------
    in_time : list or np.ndarray
        DESCRIPTION.
    in_signal : list or np.ndarray
        DESCRIPTION.
    new_time : list or np.ndarray
        DESCRIPTION.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    if (
        not isinstance(in_time, (list, np.ndarray))
        or not isinstance(in_signal, (list, np.ndarray))
        or not isinstance(new_time, (list, np.ndarray))
    ):
        raise ValueError("Input should be in the form of lists or np,array")

    if isinstance(in_time, list):
        in_time = np.ndarray(in_time)
    if isinstance(in_signal, list):
        in_signal = np.ndarray(in_signal)
    if isinstance(new_time, list):
        new_time = np.ndarray(new_time)

    set_interp = interp1d(in_time, in_signal, kind="cubic")
    y = set_interp(new_time)
    return y
    # return np.interp(new_time, in_time, in_signal, left = np.nan, right = np.nan)


def test_doctstrings():
    import doctest

    doctest.testmod()


if __name__ == "__main__":
    test_doctstrings()
