import numpy as np
from numpy import float64
from numpy.typing import NDArray

import deltares_wave_toolbox.cores.core_engine as core_engine


# --- code
def disper(w: NDArray[float64], h: float, g: float = 9.81):
    """
    DISPER  Solves the wave number from the linear dispersion relation
    The linear dispersion relation reads:

    w^2 = g * k * tanh( k * h )

    where w is the radial frequency, g the gravitation acceleration
    constant, k the wave number, and h the water depth. The function
    DISPER solves for the wave number k, given the radial frequency w,
    the water depth h and the gravitational acceleration constant g.

    water depth, units meter
         Three possible options:
         (i)   0 < h < infinity: h is water depth, and nonlinear dispersion
               relation is solved using given value for h
         (ii)  h < 0. The shallow water dispersion relation is solved, with
               the actual water depth being equal to |h|
         (iii) h = inf. The deep water dispersion relation is solved.

    This function is originally written by G. Klopman, Delft Hydraulics, 6
    Dec 1994. It is stated that the relative error in k*h < 2.5e-16 for
    all k*h

    Args:
        w (NDArray[float64]): radial frequency ( = 2 * pi / wave period), units rad/s.
        h (float): water depth, units meter.
        g (float, optional): representing gravitational constant. Defaults to 9.81.

    Returns:
        NDArray[float64]: wavenumber
    """

    # check if radial frequency is single value or of type array, if of type single value
    # convert to array to ensure this function can handle single values as well as arrays.
    w = core_engine.convert_to_array_type(w)

    # if not (isinstance(w,np.ndarray)):
    #   w = w*np.ones(1)

    if np.isinf(h):
        # --- Deep water dispersion relation
        # k = w.^2 / g
        k = w**2 / g  # element wise multiplication.
    elif h < 0:
        # --- Shallow water dispersion relation, with depth equal to |h|
        # k = w ./ sqrt( g * abs(h) );
        k = w / np.sqrt(g * abs(h))
    else:
        # --- Standard (nonlinear) dispersion relation for depths 0 < h < inf
        # w2 = (w.^2) .* h ./ g
        w2 = (w**2) * h / g

        # ielem = find( w2 < 1E-8 )
        ielem = np.nonzero(w2 < 1.0e-8)
        w2[ielem] = 1e-8
        # q  = w2 ./ (1 - exp (-(w2.^(5/4)))) .^ (2/5);
        q = w2 / (1 - np.exp(-(w2 ** (5 / 4)))) ** (2 / 5)

        idxs = [1, 2]
        for j in idxs:
            thq = np.tanh(q)
            thq2 = 1 - thq**2
            a = (1 - q * thq) * thq2
            b = thq + q * thq2
            c = q * thq - w2
            arg = np.zeros(np.size(q))
            # iq      = find (a ~= 0)
            iq = np.where(a != 0)[0]
            arg[iq] = (b[iq] ** 2) - 4 * a[iq] * c[iq]
            arg[iq] = (-b[iq] + np.sqrt(arg[iq])) / (2 * a[iq])
            iq = np.nonzero(abs(a * c) < 1.0e-8 * (b**2))
            arg[iq] = -c[iq] / b[iq]
            q = q + arg

    k = np.sign(w) * q / h

    ik = np.isnan(k)
    k[ik] = np.zeros(np.size(k[ik]))

    return k
