import matplotlib.pyplot as plt
import numpy as np
import pytest

import deltares_wave_toolbox.cores.core_dispersion as core_dispersion
import deltares_wave_toolbox.cores.core_wave_decomposition as core_wave_decomposition

t = np.linspace(0, 3600, 36000)
x = np.array([0, 0.4, 0.6, 1.2, 1.4])

# peak period
fp = [0.6, 0.7, 1.6]
# amplitudes
A = [0.1, 0.05, 0.05]
# reflection coef.
R = 0.7
# water depth
h = 0.6
# create series
xTime = np.zeros((len(t), len(x)))
xTime_in = np.zeros((len(t), len(x)))
xTime_re = np.zeros((len(t), len(x)))
for ii in range(len(x)):
    for kk in range(len(fp)):
        k = core_dispersion.disper(2 * np.pi * fp[kk], h, g=9.81)
        xTime[:, ii] = (
            xTime[:, ii]
            + A[kk] * np.cos(2 * np.pi * fp[kk] * t - k * x[ii])
            + R * A[kk] * np.cos(2 * np.pi * fp[kk] * t + k * x[ii])
        )
        xTime_in[:, ii] = xTime_in[:, ii] + A[kk] * np.cos(
            2 * np.pi * fp[kk] * t - k * x[ii]
        )
        xTime_re[:, ii] = xTime_re[:, ii] + R * A[kk] * np.cos(
            2 * np.pi * fp[kk] * t + k * x[ii]
        )


@pytest.mark.parametrize(
    ("t", "xTime", "x", "h", "xTime_in", "xTime_re"),
    (
        (t, xTime, x, h, xTime_in, xTime_re),
        (t, xTime, x, h, xTime_in, xTime_re),
    ),
)
def test_lin_ZS(t, xTime, x, h, xTime_in, xTime_re, plot_fig=False):

    xTimeIn, xTimeRe = core_wave_decomposition.decompose_linear_ZS(
        t, xTime, h, x, np.ones_like(x), detLim=0.125
    )

    id_wanted = np.logical_and(t > 60, t < 3600 - 60)
    error_in = xTimeIn[id_wanted] - xTime_in[id_wanted, 0]
    error_re = xTimeRe[id_wanted] - xTime_re[id_wanted, 0]

    assert np.max(np.abs(error_in)) == pytest.approx(0, abs=1e-3)
    assert np.max(np.abs(error_re)) == pytest.approx(0, abs=1e-3)

    if plot_fig:
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(t, xTime_in[:, 0], "k")
        plt.plot(t, xTimeIn, "r--")
        plt.legend(["Theory", "reflecNonlinear"])
        plt.ylabel(r"$\eta_{in}$")
        plt.xlabel("time [s]")
        plt.xlim([460, 620])
        plt.grid("on")
        plt.subplot(2, 1, 2)
        plt.plot(t, xTimeIn - xTime_in[:, 0])
        plt.plot(t, xTimeRe - xTime_re[:, 0])
        plt.legend(["In", "Re"])
        plt.ylabel(r"$\Delta \eta$")
        plt.xlabel("time [s]")
        plt.grid("on")
