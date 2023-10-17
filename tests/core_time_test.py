import pytest
import deltares_wave_toolbox.cores.core_time


def test_determine_zero_crossings_t1():
    # === Test 1 =============================================================
    # Short synthetic signal, with some 'nasty' situations
    # --- Test input
    t = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    x = [
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
    ]

    # --- Correct answers to tests
    nWaveDownCorrect = 3
    tCrossDownCorrect = [0.1, 0.375, 0.7, 1.2714]
    nWaveUpCorrect = 2
    tCrossUpCorrect = [0.24, 1.1, 1.34]

    # --- Call to function to compute zero-crossings
    [nWaveDown, tCrossDown] = core_time.determine_zero_crossing(t, x, "down")

    assert np.testing.assert_equal(nWaveDown, nWaveDownCorrect)

    assert np.testing.assert_allclose(
        tCrossDownCorrect, tCrossDown, rtol=0, atol=1.0e-4
    )

    [nWaveUp, tCrossUp] = core_time.determine_zero_crossing(t, x, "up")

    assert np.testing.assert_equal(nWaveUp, nWaveUpCorrect)

    assert np.testing.assert_allclose(tCrossUpCorrect, tCrossUp, rtol=0, atol=1.0e-4)
