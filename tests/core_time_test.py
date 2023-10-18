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
        (  # Short synthetic signal, with some 'nasty' situations
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
        (  # Signal constant and zero in time
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
        (  # Signal with only one up zero-crossing
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

    assert tCrossDown == pytest.approx(tCrossDownCorrect, abs=1e-4)

    assert nWaveUp == nWaveUpCorrect

    assert tCrossUp == pytest.approx(tCrossUpCorrect, abs=1e-4)


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

    assert hFracP == pytest.approx(hFracPCorrect, abs=1e-4)

    assert tFracP == pytest.approx(tFracPCorrect, abs=1e-4)


@pytest.mark.parametrize(
    ("hWaveSorted", "excPerc", "hExcPercCorrect"),
    (
        ([7.1, 6.0, 5.8, 4.4, 3.1, 2.2, 1.4], 33.0, 6.0),
        ([7.1, 6.0, 5.8, 4.4, 3.1, 2.2, 1.4], 10.0, None),
    ),
)
def test_exceedance_wave_height(hWaveSorted, excPerc, hExcPercCorrect):
    hExcPerc = core_time.exceedance_wave_height(hWaveSorted, excPerc)

    assert hExcPerc == pytest.approx(hExcPercCorrect, abs=1e-4)


@pytest.mark.parametrize(
    (
        "t_input",
        "x_input",
        "tCrossUp",
        "tCrossDown",
        "tWaveUpCorrect",
        "hWaveUpCorrect",
        "aCrestUpCorrect",
        "aTroughUpCorrect",
        "tCrestUpCorrect",
        "tTroughUpCorrect",
        "tWaveDownCorrect",
        "hWaveDownCorrect",
        "aCrestDownCorrect",
        "aTroughDownCorrect",
        "tCrestDownCorrect",
        "tTroughDownCorrect",
    ),
    (
        (  # a small synthetic signal (which includes some 'nasty' elements)
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
                -0.2,
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
            [0.2400, 1.1000, 1.3400],
            [0.1000, 0.3600, 0.7000, 1.2714],
            [0.8600, 0.2400],
            [0.5000, 0.7000],
            [0.3000, 0.5000],
            [-0.2000, -0.2000],
            [0.3000, 1.2000],
            [0.4000, 1.3000],
            [0.2600, 0.3400, 0.5714],
            [0.5000, 0.2000, 0.7000],
            [0.3000, 0.0000, 0.5000],
            [-0.2000, -0.2000, -0.2000],
            [0.3000, 0.5000, 1.2000],
            [0.2000, 0.4000, 0.8000],
        ),
        (  # A signal with only one zero crossing (hence, zero waves)
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            [-1.0, -0.5, -0.1, 0.1, 0.3, 0.6],
            [0.25],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        ),
    ),
)
def test_determine_params_individual_waves(
    t_input,
    x_input,
    tCrossUp,
    tCrossDown,
    tWaveUpCorrect,
    hWaveUpCorrect,
    aCrestUpCorrect,
    aTroughUpCorrect,
    tCrestUpCorrect,
    tTroughUpCorrect,
    tWaveDownCorrect,
    hWaveDownCorrect,
    aCrestDownCorrect,
    aTroughDownCorrect,
    tCrestDownCorrect,
    tTroughDownCorrect,
):
    [
        hWaveUp,
        tWaveUp,
        aCrestUp,
        aTroughUp,
        tCrestUp,
        tTroughUp,
    ] = core_time.determine_params_individual_waves(tCrossUp, t_input, x_input)

    assert hWaveUp == pytest.approx(hWaveUpCorrect, abs=1e-4)

    assert tWaveUp == pytest.approx(tWaveUpCorrect, abs=1e-4)

    assert aCrestUp == pytest.approx(aCrestUpCorrect, abs=1e-4)

    assert aTroughUp == pytest.approx(aTroughUpCorrect, abs=1e-4)

    assert tCrestUp == pytest.approx(tCrestUpCorrect, abs=1e-4)

    assert tTroughUp == pytest.approx(tTroughUpCorrect, abs=1e-4)

    [
        hWaveDown,
        tWaveDown,
        aCrestDown,
        aTroughDown,
        tCrestDown,
        tTroughDown,
    ] = core_time.determine_params_individual_waves(tCrossDown, t_input, x_input)

    assert hWaveDown == pytest.approx(hWaveDownCorrect, abs=1e-4)

    assert tWaveDown == pytest.approx(tWaveDownCorrect, abs=1e-4)

    assert aCrestDown == pytest.approx(aCrestDownCorrect, abs=1e-4)

    assert aTroughDown == pytest.approx(aTroughDownCorrect, abs=1e-4)

    assert tCrestDown == pytest.approx(tCrestDownCorrect, abs=1e-4)

    assert tTroughDown == pytest.approx(tTroughDownCorrect, abs=1e-4)
