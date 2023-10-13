import numpy as np
import pytest

from deltares_wave_toolbox.cores.core_wavefunctions import create_spectrum_jonswap


@pytest.fixture(scope="package")
def Hm0():
    """Hm0 value to be used in tests."""

    Hm0 = 2.0

    return Hm0


@pytest.fixture(scope="package")
def wave_spectrum(Hm0):
    """Generate wave spectrum to be used in tests."""

    # create JONSWAP Spectrum
    ff = np.linspace(0.01, 2, 1000)

    spectrum = create_spectrum_jonswap(f=ff, fp=0.1, hm0=Hm0)

    return spectrum


@pytest.fixture(scope="package")
def wave_timeseries(wave_spectrum):
    """Generate wave timeseries to be used in tests."""

    # create Series from spectrum
    timeseries = wave_spectrum.create_series(10, 3600, 0.1)

    return timeseries
