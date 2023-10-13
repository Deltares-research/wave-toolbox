import pytest


def test_get_Hm0(Hm0, wave_spectrum):
    """Test the get_Hm0 function."""

    assert wave_spectrum.get_Hm0() == pytest.approx(Hm0)
