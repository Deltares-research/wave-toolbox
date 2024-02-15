import pytest

import deltares_wave_toolbox.cores.core_dispersion as core_dispersion


@pytest.mark.parametrize(
    ("w", "h", "k_target"),
    (
        ([0.2, 0.3, 0.4, 0.5], 20, [0.0145, 0.0221, 0.0302, 0.0390]),
        ([0.2, 0.3, 0.4, 0.5], 10, [0.0203, 0.0308, 0.0415, 0.0527]),
    ),
)
def test_dispersion(w, h, k_target):
    print(w)
    k = core_dispersion.disper(w, h)

    assert k_target == pytest.approx(k, abs=1e-4)
