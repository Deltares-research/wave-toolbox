# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import figure
from numpy import float64
from numpy.typing import NDArray

import deltares_wave_toolbox.cores.core_engine as core_engine
import deltares_wave_toolbox.cores.core_spectral as core_spectral
import deltares_wave_toolbox.cores.core_wavefunctions as core_wavefunctions
import deltares_wave_toolbox.series as series


class Spectrum:
    """The wave Spectrum class

    Parameters
    ----------
    f : NDArray[float64]
        array with frequencies
    sVarDens : NDArray[float64]
        array with energy density
    D : NDArray[float64], optional
        array with directions for 2D spectrum, by default np.empty((0, 0))
    """

    def __init__(
        self,
        f: NDArray[float64],
        sVarDens: NDArray[float64],
        D: NDArray[float64] = np.empty((0, 0)),
    ) -> None:
        f, fSize = core_engine.convert_to_vector(f)
        sVarDens, SSize = core_engine.convert_to_vector(sVarDens)

        assert core_engine.monotonic_increasing_constant_step(
            f
        ), "compute_spectrum_params: Input error: frequency input parameter must be monotonic with constant step size"
        assert (
            fSize[0] == SSize[0]
        ), "compute_spectrum_params: Input error: array sizes differ in dimension"

        self.f = f
        self.sVarDens = sVarDens
        self.D = D
        self.nf = len(f)

        # 1D or 2D spectrum
        if self.D.size == 0:
            self.spec = "1D"
        else:
            self.spec = "2D"

    def __str__(self) -> str:
        return f"{self.spec} wave spectrum with {self.nf} number of frequencies"

    def __repr__(self) -> str:
        return f"{type(self).__name__} (spec = {self.spec})"

    def _set_flim(self, fmin: float = -1.0, fmax: float = -1.0) -> tuple[float, float]:
        """Set frequency limits

        Args:
            fmin (float): Minimum frequency
            fmax (float): Maximum frequency

        Returns:
            float: minimum and maximum frequency
        """
        if fmin == -1.0:
            fmin = self.f[0]
        if fmax == -1.0:
            fmax = self.f[-1]
        return fmin, fmax

    def get_Hm0(self, fmin: float = -1.0, fmax: float = -1.0) -> float:
        """Compute Hm0 of spectrum


        Parameters
        ----------
        fmin : float, optional
            Minimum frequency, by default -1.0
        fmax : float, optional
            Maximum frequency, by default -1.0

        Returns
        -------
        float
            Hm0
        """
        fmin, fmax = self._set_flim(fmin, fmax)
        m0 = core_wavefunctions.compute_moment(self.f, self.sVarDens, 0, fmin, fmax)
        self.Hm0 = 4 * np.sqrt(m0)
        return self.Hm0

    def get_Hm0_HF(self, fmin: float = -1.0, fmax: float = -1.0) -> float:
        """Compute Hm0 of the high frequency part of the spectrum

        By default, the high frequency part is defined as frequencies above
        f_cutoff = 0.45 / Tm-1,0 following De Ridder et al. (2024). For
        JONSWAP spectra, this equals a cutoff frequency of fp/2, but is a
        more stable measure for spectra without a clear peak frequency.


        Parameters
        ----------
        fmin : float, optional
            Minimum frequency, by default -1.0
        fmax : float, optional
            Maximum frequency, by default -1.0

        Returns
        -------
        float
            Hm0
        """
        if fmin == -1.0:
            if not hasattr(self, "Tmm10"):
                self.get_Tmm10()
            fmin = 0.45 / self.Tmm10

            warnings.warn(
                "Cutoff frequency fmin not set, using default value of 0.45 / Tmm10",
                UserWarning,
            )

        fmin, fmax = self._set_flim(fmin, fmax)
        m0 = core_wavefunctions.compute_moment(self.f, self.sVarDens, 0, fmin, fmax)
        self.Hm0_HF = 4 * np.sqrt(m0)
        return self.Hm0_HF

    def get_Hm0_LF(self, fmin: float = -1.0, fmax: float = -1.0) -> float:
        """Compute Hm0 of the low frequency part of the spectrum

        By default, the low frequency part is defined as frequencies below
        f_cutoff = 0.45 / Tm-1,0 following De Ridder et al. (2024). For
        JONSWAP spectra, this equals a cutoff frequency of fp/2, but is a
        more stable measure for spectra without a clear peak frequency.


        Parameters
        ----------
        fmin : float, optional
            Minimum frequency, by default -1.0
        fmax : float, optional
            Maximum frequency, by default -1.0

        Returns
        -------
        float
            Hm0
        """
        if fmax == -1.0:
            if not hasattr(self, "Tmm10"):
                self.get_Tmm10()
            fmax = 0.45 / self.Tmm10

            warnings.warn(
                "Cutoff frequency fmax not set, using default value of 0.45 / Tmm10",
                UserWarning,
            )

        fmin, fmax = self._set_flim(fmin, fmax)
        m0 = core_wavefunctions.compute_moment(self.f, self.sVarDens, 0, fmin, fmax)
        self.Hm0_LF = 4 * np.sqrt(m0)
        return self.Hm0_LF

    def get_Tps(self, fmin: float = -1.0, fmax: float = -1.0) -> float:
        """Compute Tps (smoothed peak period) of spectrum

        Parameters
        ----------
        fmin : float, optional
            Minimum frequency, by default -1.0
        fmax : float, optional
            Maximum frequency, by default -1.0

        Returns
        -------
        float
            Tps
        """
        fmin, fmax = self._set_flim(fmin, fmax)
        iFmin = np.where(self.f >= fmin)[0][0]
        iFmax = np.where(self.f <= fmax)[0][-1]
        fMiMa = self.f[iFmin : iFmax + 1]
        SMiMa = self.sVarDens[iFmin : iFmax + 1]
        self.Tps = core_wavefunctions.compute_tps(fMiMa, SMiMa)
        return self.Tps

    def get_Tp(self, fmin: float = -1.0, fmax: float = -1.0) -> float:
        """Compute Tp (peak period) of spectrum

        Parameters
        ----------
        fmin : float, optional
            Minimum frequency, by default -1.0
        fmax : float, optional
            Maximum frequency, by default -1.0

        Returns
        -------
        float
            Tp
        """
        # --- Make separate arrays containing only part corresponding to
        #     frequencies between fmin and fmax
        fmin, fmax = self._set_flim(fmin, fmax)
        iFmin = np.where(self.f >= fmin)[0][0]
        iFmax = np.where(self.f <= fmax)[0][-1]
        fMiMa = self.f[iFmin : iFmax + 1]
        SMiMa = self.sVarDens[iFmin : iFmax + 1]
        # --- Compute peak period -----------------------------------------------
        Smax = max(SMiMa)
        imax = np.where(SMiMa == Smax)[0]
        imax = imax.astype(int)
        ifp = max(imax)
        fp = fMiMa[ifp]
        #
        if np.all(ifp is None):
            ifp = 1
            fp = fMiMa[ifp]
        self.Tp = 1 / fp
        return self.Tp

    def get_Tmm10(self, fmin: float = -1.0, fmax: float = -1.0) -> float:
        """Compute Tm-1,0 spectral period of spectrum

        Parameters
        ----------
        fmin : float, optional
            Minimum frequency, by default -1.0
        fmax : float, optional
            Maximum frequency, by default -1.0

        Returns
        -------
        float
            Tmm10
        """
        fmin, fmax = self._set_flim(fmin, fmax)
        m_1 = core_wavefunctions.compute_moment(self.f, self.sVarDens, -1, fmin, fmax)
        m0 = core_wavefunctions.compute_moment(self.f, self.sVarDens, 0, fmin, fmax)
        self.Tmm10 = m_1 / m0
        return self.Tmm10

    def get_Tmm10_HF(self, fmin: float = -1.0, fmax: float = -1.0) -> float:
        """Compute Tm-1,0 spectral period for the high frequency part of the spectrum

        By default, the high frequency part is defined as frequencies above
        f_cutoff = 0.45 / Tm-1,0 following De Ridder et al. (2024). For
        JONSWAP spectra, this equals a cutoff frequency of fp/2, but is a
        more stable measure for spectra without a clear peak frequency.

        Parameters
        ----------
        fmin : float, optional
            Minimum frequency, by default -1.0
        fmax : float, optional
            Maximum frequency, by default -1.0

        Returns
        -------
        float
            Tmm10
        """
        if fmin == -1.0:
            if not hasattr(self, "Tmm10"):
                self.get_Tmm10()
            fmin = 0.45 / self.Tmm10

            warnings.warn(
                "Cutoff frequency fmin not set, using default value of 0.45 / Tmm10",
                UserWarning,
            )

        fmin, fmax = self._set_flim(fmin, fmax)
        m_1 = core_wavefunctions.compute_moment(self.f, self.sVarDens, -1, fmin, fmax)
        m0 = core_wavefunctions.compute_moment(self.f, self.sVarDens, 0, fmin, fmax)
        self.Tmm10_HF = m_1 / m0
        return self.Tmm10_HF

    def get_Tmm10_LF(self, fmin: float = -1.0, fmax: float = -1.0) -> float:
        """Compute Tm-1,0 spectral period for the low frequency part of the spectrum

        By default, the low frequency part is defined as frequencies below
        f_cutoff = 0.45 / Tm-1,0 following De Ridder et al. (2024). For
        JONSWAP spectra, this equals a cutoff frequency of fp/2, but is a
        more stable measure for spectra without a clear peak frequency.

        Parameters
        ----------
        fmin : float, optional
            Minimum frequency, by default -1.0
        fmax : float, optional
            Maximum frequency, by default -1.0

        Returns
        -------
        float
            Tmm10
        """
        if fmax == -1.0:
            if not hasattr(self, "Tmm10"):
                self.get_Tmm10()
            fmax = 0.45 / self.Tmm10

            warnings.warn(
                "Cutoff frequency fmax not set, using default value of 0.45 / Tmm10",
                UserWarning,
            )

        fmin, fmax = self._set_flim(fmin, fmax)
        m_1 = core_wavefunctions.compute_moment(self.f, self.sVarDens, -1, fmin, fmax)
        m0 = core_wavefunctions.compute_moment(self.f, self.sVarDens, 0, fmin, fmax)
        self.Tmm10_LF = m_1 / m0
        return self.Tmm10_LF

    def get_Tm01(self, fmin: float = -1.0, fmax: float = -1.0) -> float:
        """Compute Tm01 of spectrum

        Parameters
        ----------
        fmin : float, optional
            Minimum frequency, by default -1.0
        fmax : float, optional
            Maximum frequency, by default -1.0

        Returns
        -------
        float
            Tm01
        """
        fmin, fmax = self._set_flim(fmin, fmax)
        m0 = core_wavefunctions.compute_moment(self.f, self.sVarDens, 0, fmin, fmax)
        m1 = core_wavefunctions.compute_moment(self.f, self.sVarDens, 1, fmin, fmax)
        self.Tm01 = m0 / m1
        return self.Tm01

    def get_Tm02(self, fmin: float = -1.0, fmax: float = -1.0) -> float:
        """Compute Tm02 of spectrum

        Parameters
        ----------
        fmin : float, optional
            Minimum frequency, by default -1.0
        fmax : float, optional
            Maximum frequency, by default -1.0

        Returns
        -------
        float
            Tm02
        """
        fmin, fmax = self._set_flim(fmin, fmax)
        m0 = core_wavefunctions.compute_moment(self.f, self.sVarDens, 0, fmin, fmax)
        m2 = core_wavefunctions.compute_moment(self.f, self.sVarDens, 2, fmin, fmax)
        self.Tm02 = np.sqrt(m0 / m2)
        return self.Tm02

    def get_s0p(self, fmin: float = -1.0, fmax: float = -1.0, g: float = 9.81) -> float:
        """Compute s0p of the spectrum

        Compute the wave steepness of the spectrum based on the deep water wave length
        using the peak wave period Tps.

        Parameters
        ----------
        fmin : float, optional
            Minimum frequency, by default -1.0
        fmax : float, optional
            Maximum frequency, by default -1.0
        g : float, optional
            gravitational acceleration constant, by default 9.81

        Returns
        -------
        float
            s0p
        """
        if not hasattr(self, "Hm0"):
            self.get_Hm0(fmin=fmin, fmax=fmax)

        if not hasattr(self, "Tps"):
            self.get_Tps(fmin=fmin, fmax=fmax)

        self.s0p = 2 * np.pi * self.Hm0 / (g * self.Tps**2)
        return self.s0p

    def get_smm10(
        self, fmin: float = -1.0, fmax: float = -1.0, g: float = 9.81
    ) -> float:
        """Compute sm-1,0 of the spectrum

        Compute the wave steepness of the spectrum based on the deep water wave length
        using the spectral wave period Tm-1,0.

        Parameters
        ----------
        fmin : float, optional
            Minimum frequency, by default -1.0
        fmax : float, optional
            Maximum frequency, by default -1.0
        g : float, optional
            gravitational acceleration constant, by default 9.81

        Returns
        -------
        float
            smm10
        """
        if not hasattr(self, "Hm0"):
            self.get_Hm0(fmin=fmin, fmax=fmax)

        if not hasattr(self, "Tmm10"):
            self.get_Tmm10(fmin=fmin, fmax=fmax)

        self.smm10 = 2 * np.pi * self.Hm0 / (g * self.Tmm10**2)
        return self.smm10

    def get_smm10_HF(
        self, fmin: float = -1.0, fmax: float = -1.0, g: float = 9.81
    ) -> float:
        """Compute sm-1,0_HF of the spectrum

        Compute the wave steepness of the high frequency part of the
        spectrum based on the deep water wave length using the spectral
        wave period Tm-1,0_HF and the Hm0_HF.

        Parameters
        ----------
        fmin : float, optional
            Minimum frequency, by default -1.0
        fmax : float, optional
            Maximum frequency, by default -1.0
        g : float, optional
            gravitational acceleration constant, by default 9.81

        Returns
        -------
        float
            smm10_HF
        """
        if not hasattr(self, "Hm0_HF"):
            self.get_Hm0_HF(fmin=fmin, fmax=fmax)

        if not hasattr(self, "Tmm10_HF"):
            self.get_Tmm10_HF(fmin=fmin, fmax=fmax)

        self.smm10_HF = 2 * np.pi * self.Hm0_HF / (g * self.Tmm10_HF**2)
        return self.smm10_HF

    def create_series(self, tstart: float, tend: float, dt: float) -> series.Series:
        """Construct series from Spectrum with random phase

        Parameters
        ----------
        tstart : float
            Start time of time series
        tend : float
            End time of time series
        dt : float
            Time step

        Returns
        -------
        series.Series
            Series class with time series
        """
        series = core_spectral.spectrum2timeseries_object(
            self.f, self.sVarDens, tstart, tend, dt
        )
        return series

    def plot(
        self,
        savepath: str = "",
        plot_periods: bool = True,
        xlim: float = -999.0,
        xlabel: str = "f [$Hz$]",
        ylabel: str = "S [$m^2/Hz$]",
    ) -> figure.Figure:
        """Plot spectrum

        Parameters
        ----------
        savepath : str, optional
            path to save figure, by default ""
        plot_periods : bool, optional
            show different periods in the plot, by default True
        xlim : float, optional
            limit the extent of the x-axis (frequency), by default -999.0
        xlabel : str, optional
            xlabel for the plot, by default "f [$Hz$]"
        ylabel : str, optional
            ylabel for the plot, by default "f [$Hz$]"

        Returns
        -------
        figure.Figure
            figure object
        """
        fig = plt.figure()
        plt.plot(self.f, self.sVarDens, label="Spectrum")

        if plot_periods:
            if hasattr(self, "Tps"):
                plt.plot(
                    [1 / self.Tps, 1 / self.Tps],
                    [0, np.max(self.sVarDens)],
                    label="Tps",
                )
            if hasattr(self, "Tmm10"):
                plt.plot(
                    [1 / self.Tmm10, 1 / self.Tmm10],
                    [0, np.max(self.sVarDens)],
                    label="Tmm10",
                )
            if hasattr(self, "Tm02"):
                plt.plot(
                    [1 / self.Tm02, 1 / self.Tm02],
                    [0, np.max(self.sVarDens)],
                    label="Tm02",
                )
        plt.grid("on")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        if xlim != -999.0:
            plt.xlim(xlim)
        if savepath != "":
            plt.savefig(savepath)
        return fig
