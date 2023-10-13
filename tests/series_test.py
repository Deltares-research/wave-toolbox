import numpy as np
import pytest


def test_get_H2p_Rayleigh(wave_spectrum, wave_timeseries):
    """Test the get_H2p_Rayleigh function."""

    H2p = wave_timeseries.get_Hs()[0] * np.sqrt(-np.log(0.02)) / np.sqrt(2)

    assert wave_timeseries.get_H2p_Rayleigh() == pytest.approx(H2p)
