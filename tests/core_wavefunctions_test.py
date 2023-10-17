import numpy as np
import pytest

import deltares_wave_toolbox.cores.core_wavefunctions as core_wavefunctions


@pytest.mark.parametrize(
    ("f", "S", "Tps_exact"),
    (
        (0.5, 1, 2.0),
        ([1, 2], [1, 1], 1.0 / 1.5),
        ([1, 2], [1, 2], 0.5),
        ([1, 2], [2, 1], 1.0),
        ([0.5, 0.55, 0.61], [1, 0.5, 0.7], 2.0),
        ([0.35, 0.45, 0.5], [0.7, 0.5, 1], 2.0),
        ([0.35, 0.4, 0.6], [2, 1, 2], 1.0 / 0.35),
        ([0.35, 0.45, 0.5], [1, 1, 1], 1.0 / 0.45),
    ),
)
def test_compute_tps_t1(f, S, Tps_exact):
    Tps_num = core_wavefunctions.compute_tps(f, S)
    assert Tps_num == Tps_exact
