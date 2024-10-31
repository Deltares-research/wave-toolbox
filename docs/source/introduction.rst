.. highlight:: shell

============
Introduction
============


Purpose
=======

The Deltares Wave Toolbox package is made to process and generate free surface signals. Both data from physical model experiments, numerical models or synthetic generated (wave) signals can be analysed with this toolbox. The package supports analyzed in both time (using the Series and WaveHeights classes) and spectral (using the Spectrum class) domains. Common analyses available in the package include among others: zero-crossing analysis, wave height exceedance curves, converting water level elevation time series to variance density spectrum and vice versa, spectral wave parameters, theoretical spectra and wave height exceedance curves.

Structure
=========

The main structure in the package are three classes: 

* :class:`Spectrum` containing the variance density spectrum of the water level elevation
* :class:`Series` containing the time series of water level elevation, such as those obtained by wave gauges
* :class:`WaveHeights` contains the result of a zero-crossing analysis of a time series (:class:`Series` inherits from :class:`WaveHeights`)

Most users will use the :class:`Spectrum` and :class:`Series` classes. The :class:`WaveHeights` class is used internally by the :class:`Series` class. To allow for maximum flexibility however, the underlying functionality for time series and spectral analysis is included in the cores subpackage. This means that these functions can also be called independently from the aforementioned classes.

The package also enables wave decomposition into incoming and reflected waves through the :func:`decompose_linear_ZS_series` function, which is available in the :mod:`core_wave_decomposition` subpackage.
