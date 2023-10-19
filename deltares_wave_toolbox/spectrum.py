from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

import deltares_wave_toolbox.cores.core_engine as core_engine
import deltares_wave_toolbox.cores.core_spectral as core_spectral
import deltares_wave_toolbox.cores.core_wavefunctions as core_wavefunctions
import deltares_wave_toolbox.series as series


class Spectrum:
    """
    The wave Spectrum class
    """

    def __init__(
        self,
        f: npt.NDArray[np.float64],
        S: npt.NDArray[np.float64],
        D: npt.NDArray[np.float64] = np.empty((0, 0)),
    ) -> None:
        """_The init function

        Args:
            f (array): array with frequencies
            S (array): array with energy density
            D (array, optional): array with directions for 2D spectrum. Defaults to None.
        """

        f, fSize = core_engine.convert_to_vector(f)
        S, SSize = core_engine.convert_to_vector(S)

        assert core_engine.monotonic_increasing_constant_step(
            f
        ), "compute_spectrum_params: Input error: frequency input parameter must be monotonic with constant step size"
        assert (
            fSize[0] == SSize[0]
        ), "compute_spectrum_params: Input error: array sizes differ in dimension"

        self.f = f
        self.S = S
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

        Args:
            fmin (float, optional): Minimum frequency. Defaults to None.
            fmax (float, optional): Maximum frequency. Defaults to None.

        Returns:
            float: Hm0
        """
        fmin, fmax = self._set_flim(fmin, fmax)
        m0 = core_wavefunctions.compute_moment(self.f, self.S, 0, fmin, fmax)
        self.Hm0 = 4 * np.sqrt(m0)
        return self.Hm0

    def get_Tps(self, fmin: float = -1.0, fmax: float = -1.0) -> float:
        """Compute Tps (smoothed peak period) of spectrum

        Args:
            fmin (float, optional): Minimum frequency. Defaults to None.
            fmax (float, optional): Maximum frequency. Defaults to None.

        Returns:
            float: Tps
        """
        fmin, fmax = self._set_flim(fmin, fmax)
        iFmin = np.where(self.f >= fmin)[0][0]
        iFmax = np.where(self.f <= fmax)[0][-1]
        fMiMa = self.f[iFmin : iFmax + 1]
        SMiMa = self.S[iFmin : iFmax + 1]
        self.Tps = core_wavefunctions.compute_tps(fMiMa, SMiMa)
        return self.Tps

    def get_Tp(self, fmin: float = -1.0, fmax: float = -1.0) -> float:
        """Compute Tp (peak period) of spectrum

        Args:
            fmin (float, optional): Minimum frequency. Defaults to None.
            fmax (float, optional): Maximum frequency. Defaults to None.

        Returns:
            float: Tp
        """
        # --- Make separate arrays containing only part corresponding to
        #     frequencies between fmin and fmax
        fmin, fmax = self._set_flim(fmin, fmax)
        iFmin = np.where(self.f >= fmin)[0][0]
        iFmax = np.where(self.f <= fmax)[0][-1]
        fMiMa = self.f[iFmin : iFmax + 1]
        SMiMa = self.S[iFmin : iFmax + 1]
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
        """Compute Tmm10 spectral period) of spectrum

        Args:
            fmin (float, optional): Minimum frequency. Defaults to None.
            fmax (float, optional): Maximum frequency. Defaults to None.

        Returns:
            float: Tmm10
        """
        fmin, fmax = self._set_flim(fmin, fmax)
        m_1 = core_wavefunctions.compute_moment(self.f, self.S, -1, fmin, fmax)
        m0 = core_wavefunctions.compute_moment(self.f, self.S, 0, fmin, fmax)
        self.Tmm10 = m_1 / m0
        return self.Tmm10

    def get_Tm01(self, fmin: float = -1.0, fmax: float = -1.0) -> float:
        """Compute Tm01 of spectrum

        Args:
            fmin (float, optional): Minimum frequency. Defaults to None.
            fmax (float, optional): Maximum frequency. Defaults to None.

        Returns:
            float: Tm01
        """
        fmin, fmax = self._set_flim(fmin, fmax)
        m0 = core_wavefunctions.compute_moment(self.f, self.S, 0, fmin, fmax)
        m1 = core_wavefunctions.compute_moment(self.f, self.S, 1, fmin, fmax)
        self.Tm01 = m0 / m1
        return self.Tm01

    def get_Tm02(self, fmin: float = -1.0, fmax: float = -1.0) -> float:
        """Compute Tm02 of spectrum

        Args:
            fmin (float, optional): Minimum frequency. Defaults to None.
            fmax (float, optional): Maximum frequency. Defaults to None.

        Returns:
            float: Tm02
        """
        fmin, fmax = self._set_flim(fmin, fmax)
        m0 = core_wavefunctions.compute_moment(self.f, self.S, 0, fmin, fmax)
        m2 = core_wavefunctions.compute_moment(self.f, self.S, 2, fmin, fmax)
        self.Tm02 = np.sqrt(m0 / m2)
        return self.Tm02

    def create_series(self, tstart: float, tend: float, dt: float) -> series.Series:
        """Construct series from Spectrum with random phase

        Args:
            tstart (float): Start time of time series
            tend (float): End time of time series
            dt (float): Time step

        Returns:
            object: Series class with time series
        """
        series = core_spectral.spectrum2timeseries_object(
            self.f, self.S, tstart, tend, dt
        )
        return series

    def plot(
        self,
        savepath: str = "",
        fig=None,
        plot_periods: bool = True,
        xlim: float = -999.0,
    ) -> None:
        """Plot spectrum

        Args:
            savepath (str, optional): path to save figure. Defaults to None.
            fig (figure object, optional): figure object. Defaults to None.
        """

        if fig is None:
            fig = plt.figure()
        plt.plot(self.f, self.S, label="Spectrum")

        if plot_periods:
            if hasattr(self, "Tps"):
                plt.plot([1 / self.Tps, 1 / self.Tps], [0, np.max(self.S)], label="Tps")
            if hasattr(self, "Tmm10"):
                plt.plot(
                    [1 / self.Tmm10, 1 / self.Tmm10], [0, np.max(self.S)], label="Tmm10"
                )
            if hasattr(self, "Tm02"):
                plt.plot(
                    [1 / self.Tm02, 1 / self.Tm02], [0, np.max(self.S)], label="Tm02"
                )
        plt.grid("on")
        plt.xlabel("f [$Hz$]")
        plt.ylabel("S [$m^2/Hz$]")
        plt.legend()
        if xlim != -999.0:
            plt.xlim(xlim)
        if savepath != "":
            plt.savefig(savepath)
