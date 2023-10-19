import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib import figure
from scipy.stats import rayleigh

import deltares_wave_toolbox.cores.core_engine as core_engine
import deltares_wave_toolbox.cores.core_spectral as core_spectral
import deltares_wave_toolbox.cores.core_time as core_time
import deltares_wave_toolbox.cores.core_wavefunctions as core_wavefunctions
import deltares_wave_toolbox.spectrum as spectrum


class WaveHeights:
    def __init__(
        self, hwave: npt.NDArray[np.float64], twave: npt.NDArray[np.float64]
    ) -> None:
        hwave, _ = core_engine.convert_to_vector(hwave)
        twave, _ = core_engine.convert_to_vector(twave)

        self.hwave = hwave
        self.nwave = len(hwave)
        self.twave = twave

    def __str__(self) -> str:
        return f"Series with {self.nwave} elements"

    def __repr__(self) -> str:
        return f"{type(self).__name__} (series  nt = {self.nwave})"

    def sort(self) -> None:
        """Sorts the wave height and wave period.
        The sorting is done such that in hWaveSorted the wave heights of hWave
        are sorted in descending order. This same sorting is applied to
        tWave.
        """
        self.hwave, self.twave = core_time.sort_wave_params(self.hwave, self.twave)

    def get_Hrms(self) -> float:
        return np.sqrt(np.mean(self.hwave**2))

    def get_Hmax(self) -> np.float64:
        return np.max(self.hwave)

    def get_Hs(self) -> tuple[float, float]:
        return self.highest_waves(1 / 3)

    def get_H2p_Rayleigh(self) -> float:
        return self.get_Hs()[0] * rayleigh.ppf(0.98, scale=1 / np.sqrt(2)) / np.sqrt(2)

    def get_exceedance_waveheight(self, excPerc: float) -> float:
        """
        EXCEEDANCEWAVEHEIGHT  Computes wave height with given exceedance probability

        This function computes the wave height hExcPerc with given exceedance
        probability percentage excPerc.

        Args:
            excPerc (float): exceedance probability percentage. excPerc = 2 means an
                exceedance percentage of 2%. The value of excPerc should
                not exceed 100, or be smaller than 0

        Returns:
            hExcPerc(float):  wave height with given exceedance probability
        """
        self.hwave, self.twave = core_time.sort_wave_params(self.hwave, self.twave)
        return core_time.exceedance_wave_height(hWaveSorted=self.hwave, excPerc=excPerc)

    def highest_waves(self, fracP: float) -> tuple[float, float]:
        """
        HIGHEST_WAVES_PARAMS  Computes wave parameters of selection largest waves

        This function computes the wave height hFracP and wave period tFracP by
        taking the average of the fraction fracP of the highest waves. When
        fracP = 1/3, then hFracP is equal to the significant wave height and
        tFracP is equal to the significant wave period.

        Subject: time domain analysis of waves

        Args:
            fracP       : double
                fraction. Should be between 0 and 1

        Returns:
            hFracP     : double
                    average of the wave heights of the highest fracP waves
            tFracP     : double
                    average of the wave periods of the highest fracP waves
        """
        self.hwave, self.twave = core_time.sort_wave_params(self.hwave, self.twave)
        hFracP, tFracP = core_time.highest_waves_params(
            hWaveSorted=self.hwave, tWaveSorted=self.twave, fracP=fracP
        )
        return hFracP, tFracP

    def plot_exceedance_waveheight(self, savepath: str = "") -> figure.Figure:
        """Plot exceedances of wave heights

        Args:
            savepath (str, optional): path to save figure. Defaults to None.
            fig (figure object, optional): figure object. Defaults to None.
        """
        self.hwave, self.twave = core_time.sort_wave_params(self.hwave, self.twave)
        fig = plt.figure()
        plt.plot(
            self.hwave, np.linspace(0, self.nwave, self.nwave, self.nwave) / self.nwave
        )

        plt.yscale("log")
        plt.grid("on")
        plt.xlabel("Wave height [$m$]")
        plt.ylabel("P exceedance ")
        plt.legend()

        if savepath != "":
            plt.savefig(savepath)
        return fig

    def plot_exceedance_waveheight_Rayleigh(
        self,
        savepath: str = "",
        normalize: bool = False,
        plot_BG: bool = False,
        water_depth: float = -1.0,
        cota_slope: float = 250.0,
        Hm0: float = -1.0,
    ) -> figure.Figure:
        """Plot exceedances of wave heights compared to Rayleigh distribution

        Args:
            savepath (str, optional): path to save figure. Defaults to None.
            fig (figure object, optional): figure object. Defaults to None.
        """

        if normalize:
            # Normalize with significant wave height
            H_normalize = self.get_Hs()[0]
            y_label = r"Normalized wave height $\frac{H_{i}}{H_{s}}$ [-]"
        else:
            # no normalization
            H_normalize = 1
            y_label = r"Wave height $H_{i}$ [$m$]"

        Rayleigh_x = np.sqrt(-np.log(np.arange(1, self.nwave + 1) / self.nwave))
        H2p_Rayleigh = self.get_H2p_Rayleigh()
        Rayleigh_theoretical_dist = H2p_Rayleigh * np.sqrt(
            np.log(1 - np.arange(self.nwave, 0, -1) / (self.nwave + 1)) / np.log(0.02)
        )

        self.hwave, self.twave = core_time.sort_wave_params(self.hwave, self.twave)
        fig = plt.figure()

        plt.plot(
            Rayleigh_x,
            Rayleigh_theoretical_dist / H_normalize,
            label="Theoretical Rayleigh distribution",
        )

        if plot_BG:
            if Hm0 == -1.0:
                raise ValueError(
                    "Please provide Hm0 when using Battjes & Groenendijk distribution"
                )
            else:
                (
                    hwave_BG,
                    Pexceedance_BG,
                ) = core_wavefunctions.compute_BattjesGroenendijk_wave_height_distribution(
                    Hm0, self.nwave, water_depth, cota_slope=cota_slope
                )

                plt.plot(
                    np.sqrt(-np.log(Pexceedance_BG)),
                    hwave_BG / H_normalize,
                    label="Battjes & Groenendijk (2000) distribution",
                )

        plt.plot(Rayleigh_x, self.hwave / H_normalize, label="Wave height distribution")

        plt.grid("on")
        plt.xlabel(r"$P_{exceedance}$ [$\%$]")
        plt.ylabel(y_label)
        plt.legend()

        xtick_positions = np.array([100, 50, 20, 10, 5, 2, 1, 0.1, 0.01, 0.001])
        ax = plt.gca()
        ax.set_xticks(
            rayleigh.ppf(1 - xtick_positions / 100, scale=1 / np.sqrt(2)),
            labels=xtick_positions,
        )

        if savepath != "":
            plt.savefig(savepath)
        return fig

    def plot_hist_waveheight(self, savepath: str = "") -> figure.Figure:
        """Plot Histogram of wave heights

        Args:
            savepath (str, optional): path to save figure. Defaults to None.
            fig (figure object, optional): figure object. Defaults to None.
        """

        fig = plt.figure()
        plt.hist(self.hwave, label="Distribution")

        plt.grid("on")
        plt.xlabel("H [$m$]")
        plt.ylabel("P ")
        plt.legend()

        if savepath != "":
            plt.savefig(savepath)
        return fig


class Series(WaveHeights):
    """Series object

    Args:
        WaveHeights (_type_): _description_

    Returns:
        _type_: _description_
    """

    def __init__(
        self, time: npt.NDArray[np.float64], x: npt.NDArray[np.float64]
    ) -> None:
        time, tSize = core_engine.convert_to_vector(time)
        x, xSize = core_engine.convert_to_vector(x)

        assert tSize[0] == xSize[0], "Input error: array sizes differ in dimension"

        self.time = time
        self.x = x
        self.nt = len(time)

        [
            hWave,
            tWave,
            _,
            _,
            _,
            _,
        ] = self._determine_individual_waves()
        super().__init__(hWave, tWave)

    def __str__(self) -> str:
        return f"Series with {self.nt} elements"

    def __repr__(self) -> str:
        return f"{type(self).__name__} (series  nt = {self.nt})"

    def get_crossing(
        self, typeCross: str = "down"
    ) -> tuple[int, npt.NDArray[np.float64]]:
        """Get zero crossings

        Args:
            typeCross (str, optional): type of crossing ('up' or 'down'). Defaults to 'down'.

        Returns:
            float and array: nWave with number of waves and tCross with the moments in time with a crossing
        """
        nWave, tCross = core_time.determine_zero_crossing(
            t=self.time, x=self.x, typeCross=typeCross
        )
        return nWave, tCross

    def get_spectrum(self, fres: float = 0.01) -> spectrum.Spectrum:
        """create spectrum

        Args:
            fres (float, optional): frequency resolution. Defaults to 0.01.

        Returns:
            Spectrum object: Spectrum
        """

        [f, S] = core_spectral.compute_spectrum_time_series(self.time, self.x, fres)
        return spectrum.Spectrum(f, S)

    def max(self) -> float:
        return np.max(self.x)

    def min(self) -> float:
        return np.min(self.x)

    def mean(self) -> float:
        return np.mean(self.x)

    def var(self) -> float:
        return np.var(self.x)

    def get_fourier_comp(
        self,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.complex128], bool]:
        """get Fourier components from series

        Returns:
            array: f, frequency array
            array: xFreq, fourrier components
            isOdd: logical
        """
        f, xFreq, isOdd = core_spectral.time2freq_nyquist(self.time, self.x)
        return f, xFreq, isOdd

    def _determine_individual_waves(
        self, typeCross: str = "down"
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """determine individual waves in series

        Args:
            typeCross (str, optional): type of crossing (down or up). Defaults to 'down'.

        Returns
            tWave    : array double (1D)
                    1D array containing the periods of the individual waves
            hWave    : array double (1D)
                    1D array containing the wave heights of the individual waves
            aCrest   : array double (1D)
                    1D array containing the maximum amplitudes of the crest of
                    the individual waves
            aTrough  : array double (1D)
                    1D array containing the maximum amplitudes of the trough of
                    the individual waves
            tCrest   : array double (1D)
                    1D array containing the time at which maximum crest
                    amplitude of the individual waves occurs
            tTrough  : array double (1D)
                    1D array containing the time at which maximum trough
                    amplitude of the individual waves occurs
            Notes:
                * All these arrays have a length equal to nWave, which is the number of
                waves in the wave train
                * The values of aTrough are always smaller than zero
                * hWave = aCrest - aTrough
        """
        _, tCross = core_time.determine_zero_crossing(
            t=self.time, x=self.x, typeCross=typeCross
        )
        (
            hWave,
            tWave,
            aCrest,
            aTrough,
            tCrest,
            tTrough,
        ) = core_time.determine_params_individual_waves(
            tCross=tCross, t=self.time, x=self.x
        )
        return hWave, tWave, aCrest, aTrough, tCrest, tTrough

    def plot(self, savepath: str = "", plot_crossing: bool = False) -> figure.Figure:
        """Plot Series

        Args:
            savepath (str, optional): path to save figure. Defaults to None.
            fig (figure object, optional): figure object. Defaults to None.
        """

        fig = plt.figure()
        plt.plot(self.time, self.x, label="series")
        if plot_crossing:
            _, tCross = self.get_crossing()
            plt.plot(tCross, np.asarray(tCross) * 0, "ro", label="crossing")
        plt.grid("on")
        plt.xlabel("time [$s$]")
        plt.ylabel("z [$m$]")
        plt.legend()

        if savepath != "":
            plt.savefig(savepath)
        return fig
