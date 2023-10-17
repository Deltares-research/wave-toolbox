import numpy as np
import pytest

import deltares_wave_toolbox.cores.core_time as core_time


@pytest.mark.parametrize(
    ("hWave", "tWave", "hWaveSortedCorrect", "tWaveSortedCorrect"),
    (
        ([2, 4, 3], [6, 7, 8], [4, 3, 2], [7, 8, 6]),
        (
            [3.1, 2.2, 1.4, 5.8, 7.1, 4.4, 6.0],
            [2.0, 3.0, 1.1, 3.7, 5.1, 0.0, 3.2],
            [7.1, 6.0, 5.8, 4.4, 3.1, 2.2, 1.4],
            [5.1, 3.2, 3.7, 0.0, 2.0, 3.0, 1.1],
        ),
    ),
)
def test_sort_wave_params(hWave, tWave, hWaveSortedCorrect, tWaveSortedCorrect):
    [hWaveSorted, tWaveSorted] = core_time.sort_wave_params(hWave, tWave)

    # TODO 2 assert statements in one test isn't ideal, check for workable alternatives
    assert (hWaveSorted == hWaveSortedCorrect).all()

    assert (tWaveSorted == tWaveSortedCorrect).all()


@pytest.mark.parametrize(
    (
        "t_input",
        "x_input",
        "nWaveDownCorrect",
        "tCrossDownCorrect",
        "nWaveUpCorrect",
        "tCrossUpCorrect",
    ),
    (
        (
            [
                0.0,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                1.0,
                1.1,
                1.2,
                1.3,
                1.4,
                1.5,
            ],
            [
                1.0,
                0.0,
                -0.2,
                0.3,
                -0.1,
                0.0,
                0.0,
                0.0,
                -0.2,
                0.0,
                0.0,
                0.0,
                0.5,
                -0.2,
                0.3,
                0.0,
            ],
            3,
            [0.1, 0.375, 0.7, 1.2714],
            2,
            [0.24, 1.1, 1.34],
        ),
        (
            [
                0.0,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                1.0,
                1.1,
                1.2,
                1.3,
                1.4,
                1.5,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            0,
            [],
            0,
            [],
        ),
        (
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            [-1.0, -0.5, -0.1, 0.1, 0.3, 0.6],
            0,
            [],
            0,
            [0.25],
        ),
    ),
)
def test_determine_zero_crossings(
    t_input,
    x_input,
    nWaveDownCorrect,
    tCrossDownCorrect,
    nWaveUpCorrect,
    tCrossUpCorrect,
):
    [nWaveDown, tCrossDown] = core_time.determine_zero_crossing(
        t_input, x_input, "down"
    )
    [nWaveUp, tCrossUp] = core_time.determine_zero_crossing(t_input, x_input, "up")

    # TODO 4 assert statements in one test isn't ideal, check for workable alternatives
    assert nWaveDown == nWaveDownCorrect

    # TODO fails for rel=1e-6 for pattern 1, check whether something is wrong
    assert tCrossDown == pytest.approx(tCrossDownCorrect, rel=1e-4)

    assert nWaveUp == nWaveUpCorrect

    assert tCrossUp == pytest.approx(tCrossUpCorrect)


@pytest.mark.parametrize(
    ("hWaveSorted", "tWaveSorted", "fracP", "hFracPCorrect", "tFracPCorrect"),
    (
        (
            [7.1, 6.0, 5.8, 4.4, 3.1, 2.2, 1.4],
            [5.1, 3.2, 3.7, 0.0, 2.0, 3.0, 1.1],
            1 / 3,
            6.55,
            4.15,
        ),
        (
            [7.1, 6.0, 5.8, 4.4, 3.1, 2.2, 1.4],
            [5.1, 3.2, 3.7, 0.0, 2.0, 3.0, 1.1],
            1 / 8,
            None,
            None,
        ),
    ),
)
def test_highest_waves_params(
    hWaveSorted, tWaveSorted, fracP, hFracPCorrect, tFracPCorrect
):
    [hFracP, tFracP] = core_time.highest_waves_params(hWaveSorted, tWaveSorted, fracP)

    assert hFracP == pytest.approx(hFracPCorrect)

    assert tFracP == pytest.approx(tFracPCorrect)


@pytest.mark.parametrize(
    ("hWaveSorted", "excPerc", "hExcPercCorrect"),
    (
        ([7.1, 6.0, 5.8, 4.4, 3.1, 2.2, 1.4], 33.0, 6.0),
        ([7.1, 6.0, 5.8, 4.4, 3.1, 2.2, 1.4], 10.0, None),
    ),
)
def test_exceedance_wave_height(hWaveSorted, excPerc, hExcPercCorrect):
    hExcPerc = core_time.exceedance_wave_height(hWaveSorted, excPerc)

    assert hExcPerc == pytest.approx(hExcPercCorrect)
