import pytest


def test_get_Hm0(Hm0, wave_spectrum):
    assert wave_spectrum.get_Hm0() == pytest.approx(Hm0)


# TODO fails for rel=1e-6, check whether frequency discretization isn't too coarse
def test_get_Tp(Tp, wave_spectrum):
    assert wave_spectrum.get_Tp() == pytest.approx(Tp, rel=1e-2)
