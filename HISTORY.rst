=======
History
=======

1.1.1 (2025-08-01)
------------------

* Patch to the "Rogue" release
* Added feature: the get_spectrum() method now features a dfDesired keyword, enabling specifying the desired frequency resolution (Series class)
* Changed default: by default, the get_spectrum() method uses a dfDesired of 0.01 Hz. The previous default behavior can be restored by setting use_dfDesired=False, but often leads to very course frequency resolution.


1.1.0 (2024-11-01)
------------------

* "Rogue" release
* Python 3.9 support dropped (due to issues with scipy)
* Python 3.13 support added
* Added feature: wave decomposition into incoming and reflected waves (cores.core_wave_decomposition.decompose_linear_ZS_series)
* Added feature: calculate wave steepness and high or low-frequency wave parameters (Spectrum class)

1.0.0 (2023-10-27)
------------------

* Initial "Swell" release