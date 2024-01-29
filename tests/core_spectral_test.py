import numpy as np
import pytest

import deltares_wave_toolbox.cores.core_spectral as core_spectral


@pytest.mark.parametrize(
    ("f", "xFreq", "fTotCorrect", "xFreqTotCorrect", "isOdd"),
    (
        (
            [0, 0.4, 0.8, 1.2, 1.62, 2.0],
            [
                6.3962 + 0.0j,
                0.7636 + 0.3877j,
                -1.1051 - 0.0730j,
                1.1387 - 0.9262j,
                0.5616 - 1.5596j,
                -0.0759 - 0.3736j,
            ],
            [
                0,
                0.4000,
                0.8000,
                1.2000,
                1.6000,
                2.0000,
                2.4000,
                2.8000,
                3.2000,
                3.6000,
                4.0000,
            ],
            [
                6.3962 + 0j,
                0.7636 + 0.3877j,
                -1.1051 - 0.0730j,
                1.1387 - 0.9262j,
                0.5616 - 1.5596j,
                -0.0759 - 0.3736j,
                -0.0759 + 0.3736j,
                0.5616 + 1.5596j,
                1.1387 + 0.9262j,
                -1.1051 + 0.0730j,
                0.7636 - 0.3877j,
            ],
            True,
        ),
        (
            [0, 0.5000, 1.0000, 1.5000, 2.0000, 2.5000],
            [
                6.2386 + 0j,
                1.3776 + 0.2681j,
                -0.2869 + 1.0963j,
                0.7246 - 0.5531j,
                -0.5517 - 0.8957j,
                -0.6184 + 0j,
            ],
            [0, 0.5000, 1.0000, 1.5000, 2.0000, 2.5000, 3.0000, 3.5000, 4.0000, 4.5000],
            [
                6.2386 + 0j,
                1.3776 + 0.2681j,
                -0.2869 + 1.0963j,
                0.7246 - 0.5531j,
                -0.5517 - 0.8957j,
                -0.6184 + 0j,
                -0.5517 + 0.8957j,
                0.7246 + 0.5531j,
                -0.2869 - 1.0963j,
                1.3776 - 0.2681j,
            ],
            False,
        ),
    ),
)
def test_unfold_spectrum(f, xFreq, fTotCorrect, xFreqTotCorrect, isOdd):
    fTot, xFreqTot = core_spectral.unfold_spectrum(f, xFreq, isOdd)

    # TODO 2 assert statements in one test isn't ideal, check for workable alternatives
    assert fTot == pytest.approx(fTotCorrect, abs=1e-4)

    assert xFreqTot == pytest.approx(xFreqTotCorrect, abs=1e-4)


def test_freq2time():
    # --- Create time signal
    dt = 0.1
    t = np.arange(0, 100) * dt
    a1 = 0.5
    w1 = 2 * np.pi / 5
    phi1 = 0.35  # Wave component 1
    a2 = 0.7
    w2 = 2 * np.pi / 6
    phi2 = 0.96  # Wave component 2
    y = a1 * np.cos(w1 * t + phi1) + a2 * np.cos(w2 * t + phi2)
    # --- Compute discrete Fourier transform
    f, yFreq = core_spectral.time2freq(t, y)
    yTime = core_spectral.freq2time(yFreq)

    assert yTime == pytest.approx(y, abs=1e-4)


def test_compute_spectrum_time_series():
    dt = 0.1
    t = np.arange(0, 5 + dt, dt)  # Time axis
    z = np.sin(t) + np.cos(2 * t)  # Surface elevation data
    df = 0.01
    [fS, S] = core_spectral.compute_spectrum_time_series(t, z, df)

    fCorrect = [
        0,
        0.1961,
        0.3922,
        0.5882,
        0.7843,
        0.9804,
        1.1765,
        1.3725,
        1.5686,
        1.7647,
        1.9608,
        2.1569,
        2.3529,
        2.5490,
        2.7451,
        2.9412,
        3.1373,
        3.3333,
        3.5294,
        3.7255,
        3.9216,
        4.1176,
        4.3137,
        4.5098,
        4.7059,
        4.9020,
    ]

    SCorrect = [
        0.0642,
        1.7589,
        2.6036,
        0.3509,
        0.1539,
        0.0897,
        0.0599,
        0.0434,
        0.0332,
        0.0264,
        0.0217,
        0.0183,
        0.0158,
        0.0139,
        0.0123,
        0.0111,
        0.0102,
        0.0094,
        0.0088,
        0.0083,
        0.0079,
        0.0076,
        0.0074,
        0.0072,
        0.0071,
        0.0070,
    ]

    assert fS == pytest.approx(fCorrect, abs=1e-4)

    assert S == pytest.approx(SCorrect, abs=1e-4)


@pytest.mark.parametrize(
    ("fCorrect", "SCorrect", "df"),
    (
        (
            [
                0,
                0.1961,
                0.3922,
                0.5882,
                0.7843,
                0.9804,
                1.1765,
                1.3725,
                1.5686,
                1.7647,
                1.9608,
                2.1569,
                2.3529,
                2.5490,
                2.7451,
                2.9412,
                3.1373,
                3.3333,
                3.5294,
                3.7255,
                3.9216,
                4.1176,
                4.3137,
                4.5098,
                4.7059,
                4.9020,
            ],
            [
                0.0642,
                1.7589,
                2.6036,
                0.3509,
                0.1539,
                0.0897,
                0.0599,
                0.0434,
                0.0332,
                0.0264,
                0.0217,
                0.0183,
                0.0158,
                0.0139,
                0.0123,
                0.0111,
                0.0102,
                0.0094,
                0.0088,
                0.0083,
                0.0079,
                0.0076,
                0.0074,
                0.0072,
                0.0071,
                0.0070,
            ],
            0.01,
        ),
        (
            [
                0,
                0.1961,
                0.3922,
                0.5882,
                0.7843,
                0.9804,
                1.1765,
                1.3725,
                1.5686,
                1.7647,
                1.9608,
                2.1569,
                2.3529,
                2.5490,
                2.7451,
                2.9412,
                3.1373,
                3.3333,
                3.5294,
                3.7255,
                3.9216,
                4.1176,
                4.3137,
                4.5098,
                4.7059,
                4.9020,
            ],
            [
                0.0642,
                1.7589,
                2.6036,
                0.3509,
                0.1539,
                0.0897,
                0.0599,
                0.0434,
                0.0332,
                0.0264,
                0.0217,
                0.0183,
                0.0158,
                0.0139,
                0.0123,
                0.0111,
                0.0102,
                0.0094,
                0.0088,
                0.0083,
                0.0079,
                0.0076,
                0.0074,
                0.0072,
                0.0071,
                0.0070,
            ],
            0.0,
        ),
    ),
)
def test_compute_spectrum_freq_series(fCorrect, SCorrect, df):
    dt = 0.1
    t = np.arange(0, 5 + dt, dt)  # Time axis
    z = np.sin(t) + np.cos(2 * t)  # Surface elevation data
    [f, xFreq, isOdd] = core_spectral.time2freq_nyquist(t, z)
    [fS, S] = core_spectral.compute_spectrum_freq_series(f, xFreq, len(t), df)

    assert fS == pytest.approx(fCorrect, abs=1e-4)

    assert S == pytest.approx(SCorrect, abs=1e-4)


def test_bandpassfilter():
    dt = 0.1
    t = np.arange(0, 500 + dt, dt)  # Time axis
    z = 0.5 * np.sin(t * 2 * np.pi * 0.1) + np.cos(
        t * 2 * np.pi * 0.2
    )  # Surface elevation data
    z_filter1 = core_spectral.bandpassfilter(t, z, 0, 0.15)
    z_filter2 = core_spectral.bandpassfilter(t, z, 0.15, 0.5)

    assert np.max(z_filter1) == pytest.approx(0.5, abs=0.02)

    assert np.max(z_filter2) == pytest.approx(1.0, abs=0.02)
