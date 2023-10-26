import matplotlib.pyplot as plt
import numpy as np
from matplotlib import figure
from numpy import complex128, float64
from numpy.typing import NDArray
from scipy.stats import rayleigh

import deltares_wave_toolbox.cores.core_engine as core_engine
import deltares_wave_toolbox.cores.core_spectral as core_spectral
import deltares_wave_toolbox.cores.core_time as core_time
import deltares_wave_toolbox.cores.core_wavefunctions as core_wavefunctions
import deltares_wave_toolbox.spectrum as spectrum


class WaveHeights:
    """The WaveHeights class

    Parameters
    ----------
    hwave : NDArray[float64]
        1D array containing the wave heights of the individual waves [m]
    twave : NDArray[float64]
        1D array containing the periods of the individual waves [s]

    """

    def __init__(self, hwave: NDArray[float64], twave: NDArray[float64]) -> None:
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
        """Sort wave heights and wave periods

        Sorts the wave height and wave period.
        The sorting is done such that in hWaveSorted the wave heights of hWave
        are sorted in descending order. This same sorting is applied to
        tWave.

        """
        self.hwave, self.twave = core_time.sort_wave_params(self.hwave, self.twave)

    def get_Hrms(self) -> float:
        """Compute root-mean-squared wave height

        Returns
        -------
        float
            Hrms

        """
        return np.sqrt(np.mean(self.hwave**2))

    def get_Hmax(self) -> float64:
        """Return maximum wave height

        Returns
        -------
        float64
            Hmax

        """
        return np.max(self.hwave)

    def get_Hs(self) -> tuple[float, float]:
        """Compute significant wave height & associated period

        Hs (significant wave height) is the average of the highest 1/3 of the waves

        Returns
        -------
        tuple[float, float]
            Hs, Ts

        """
        return self.highest_waves(1 / 3)

    def get_H2p_Rayleigh(self) -> float:
        """Compute 2% exceedance wave height assuming theoretical Rayleigh distribution

        Returns
        -------
        float
            H2p

        """
        return self.get_Hs()[0] * rayleigh.ppf(0.98, scale=1 / np.sqrt(2)) / np.sqrt(2)

    def get_exceedance_waveheight(self, excPerc: float) -> float:
        """Computes wave height with given exceedance probability

        This function computes the wave height hExcPerc with given exceedance probability percentage excPerc.

        Parameters
        ----------
        excPerc : float
            exceedance probability percentage. excPerc = 2 means an exceedance percentage of 2%. The value of excPerc
            should not exceed 100, or be smaller than 0 [%]

        Returns
        -------
        float
            wave height with given exceedance probability [m]

        """
        self.hwave, self.twave = core_time.sort_wave_params(self.hwave, self.twave)
        return core_time.exceedance_wave_height(hWaveSorted=self.hwave, excPerc=excPerc)

    def highest_waves(self, fracP: float) -> tuple[float, float]:
        """Computes wave parameters of selection largest waves

        This function computes the wave height hFracP and wave period tFracP by taking the average of the fraction
        fracP of the highest waves. When fracP = 1/3, then hFracP is equal to the significant wave height and tFracP
        is equal to the significant wave period.

        Parameters
        ----------
        fracP : float
            fraction. Should be between 0 and 1 [-]

        Returns
        -------
        tuple[float, float]
            hFracP : float
                average of the wave heights of the highest fracP waves [m]
            tFracP : float
                average of the wave periods of the highest fracP waves [s]

        """
        self.hwave, self.twave = core_time.sort_wave_params(self.hwave, self.twave)
        hFracP, tFracP = core_time.highest_waves_params(
            hWaveSorted=self.hwave, tWaveSorted=self.twave, fracP=fracP
        )
        return hFracP, tFracP

    def plot_exceedance_waveheight(self, savepath: str = "") -> figure.Figure:
        """Plot exceedances of wave heights

        Parameters
        ----------
        savepath : str, optional
            path to save figure, by default ""

        Returns
        -------
        figure.Figure
            figure object

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
        hm0: float = -1.0,
    ) -> figure.Figure:
        """Plot exceedances of wave heights compared to Rayleigh distribution

        Parameters
        ----------
        savepath : str, optional
            path to save figure, by default ""
        normalize : bool, optional
            normalize wave heights with significant wave height (Hs), by default False
        plot_BG : bool, optional
            include theoretical Battjes & Groenendijk (2000) distribution in plot, by default False
        water_depth : float, optional
            water depth needed for Battjes & Groenendijk (2000), by default -1.0 [m]
        cota_slope : float, optional
            foreshore slope needed for Battjes & Groenendijk (2000), by default 250.0 [-]
        hm0 : float, optional
            spectral wave height needed for Battjes & Groenendijk (2000), by default -1.0 [m]

        Returns
        -------
        figure.Figure
            figure object

        Raises
        ------
        ValueError
            Raised when plot_BG is True and Hm0 is not provided

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
            if hm0 == -1.0:
                raise ValueError(
                    "Please provide Hm0 when using Battjes & Groenendijk distribution"
                )
            else:
                (
                    hwave_BG,
                    Pexceedance_BG,
                ) = core_wavefunctions.compute_BattjesGroenendijk_wave_height_distribution(
                    hm0, self.nwave, water_depth, cota_slope=cota_slope
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

        Parameters
        ----------
        savepath : str, optional
            path to save figure, by default ""

        Returns
        -------
        figure.Figure
            figure object

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
    """The wave Series class

    Contains a time series (typically) of water level elevations. Inherits from WaveHeights class.

    Parameters
    ----------
    time : NDArray[float64]
        1D array containing time axis. The numbers in the array t must be increasing and uniformly spaced
        (uniform time step) [s]
    xTime : NDArray[float64]
        1D array containing signal values, i.e. the time series of the signal. The value xTime(i) must be the signal
        value at time t(i). Usually water surface elevation [m]

    """

    def __init__(self, time: NDArray[float64], xTime: NDArray[float64]) -> None:
        time, tSize = core_engine.convert_to_vector(time)
        xTime, xSize = core_engine.convert_to_vector(xTime)

        assert tSize[0] == xSize[0], "Input error: array sizes differ in dimension"
        assert np.var(np.diff(time)) < np.mean(
            np.diff(time) / 100
        ), "Input error: time step is not equidistant"

        self.time = time
        self.xTime = xTime
        self.nt = len(time)
        self.dt = np.mean(np.diff(self.time))

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

    def get_crossing(self, typeCross: str = "down") -> tuple[int, NDArray[float64]]:
        """Get zero crossings form time series

        Determine either the zero up- or down-crossings of the time series.

        Parameters
        ----------
        typeCross : str, optional
            Search for up- or down-crossings, by default "down"

        Returns
        -------
        tuple[int, NDArray[float64]]
            nWave : int
                Number of waves in the signal, where one wave corresponds to two successive zero-crossings. Wave i
                starts at time tCross(i), and end at time tCross(i+1) [-]
            tCross : NDArray[float64]
                1D array of length (nWave+1), containing the time of all zero-crossings. The time of the
                zero-crossings is determined by linear interpolation. Note that in case of no zero-crossing, the
                array tCross is empty. Note that in case of one zero-crossing, the number of waves is zero. [s]

        """
        nWave, tCross = core_time.determine_zero_crossing(
            t=self.time, xTime=self.xTime, typeCross=typeCross
        )
        return nWave, tCross

    def get_spectrum(
        self,
        nperseg: int = 256,
        noverlap: int = 0,
        nfft: int = 0,
        windows_type: str = "hann",
    ) -> spectrum.Spectrum:
        """create spectrum

        Parameters
        ----------
        nperseg : int, optional
            Length of each segment, by default None
        noverlap : int, optional
            number of points in overlap, by default None
        nfft : int, optional
            length of fft, by default None
        window_type : str, optional
            window type, by default "hann"

        Returns
        -------
        spectrum.Spectrum
            Spectrum in spectrum object

        """
        [f, S] = core_spectral.compute_spectrum_welch_wrapper(
            self.xTime,
            dt=self.dt,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            window_type=windows_type,
        )
        return spectrum.Spectrum(f, S)

    def get_spectrum_raw(self, dfDesired: float = 0.01) -> spectrum.Spectrum:
        """create spectrum with a desired frequency, but without applying segments

        Parameters
        ----------
        dfDesired : float, optional
            desired frequency spacing in Hertz on which the wave spectrum must be computed. If this parameter is omitted,
            then dfDesired = f(1) - f(0), by default 0.01 [Hz]

        Returns
        -------
        spectrum.Spectrum
            Spectrum in spectrum object

        """
        [f, S] = core_spectral.compute_spectrum_time_series(
            self.time, self.xTime, dfDesired
        )
        return spectrum.Spectrum(f, S)

    def max(self) -> float:
        return np.max(self.xTime)

    def min(self) -> float:
        return np.min(self.xTime)

    def mean(self) -> float:
        return np.mean(self.xTime)

    def var(self) -> float:
        return np.var(self.xTime)

    def get_fourier_comp(
        self,
    ) -> tuple[NDArray[float64], NDArray[complex128], bool]:
        """get Fourier components from series

        Returns
        -------
        tuple[NDArray[float64], NDArray[complex128], bool]
            f : NDArray[float64]
                1D array containing frequency values, for folded Fourier transform. The frequency axis runs from 0 to
                the Nyquist frequency. The number of elements in array f is close to half the number of elements in
                array xTime. To be precise, length(f) = floor(nT/2) + 1, with nT the number of elements in array
                xTime [Hz]
            xFreq : NDArray[complex128]
                1D array (complex!) containing the folded Fourier coefficients of xTime. The value xFreq(i) must be the
                Fourier coefficient at frequency f(i). The number of elements in f and xFreq are the same.
            isOdd : bool
                logical indicating whether nT, the number of time points in xTime, is even (isOdd=False) or odd
                (isOdd=True)

        """
        f, xFreq, isOdd = core_spectral.time2freq_nyquist(self.time, self.xTime)
        return f, xFreq, isOdd

    def _determine_individual_waves(
        self, typeCross: str = "down"
    ) -> tuple[
        NDArray[float64],
        NDArray[float64],
        NDArray[float64],
        NDArray[float64],
        NDArray[float64],
        NDArray[float64],
    ]:
        """determine individual waves in series

        Parameters
        ----------
        typeCross : str, optional
            Search for up- or down-crossings, by default "down"

        Returns
        -------
        tuple[ NDArray[float64], NDArray[float64], NDArray[float64], NDArray[float64], NDArray[float64],
        NDArray[float64], ]
            tWave : NDArray[float64]
                1D array containing the periods of the individual waves [s]
            hWave : NDArray[float64]
                1D array containing the wave heights of the individual waves [m]
            aCrest : NDArray[float64]
                1D array containing the maximum amplitude of the crests of the individual waves [m]
            aTrough : NDArray[float64]
                1D array containing the maximum amplitude of the troughs of the individual waves [m]
            tCrest : NDArray[float64]
                1D array containing the time at which maximum crest amplitude of the individual waves occurs [s]
            tTrough : NDArray[float64]
                1D array containing the time at which maximum trough amplitude of the individual waves occurs [s]

        Notes
        -----
        * All these arrays have a length equal to nWave, which is the number of waves in the wave train
        * The values of aTrough are always smaller than zero
        * hWave = aCrest - aTrough

        """
        _, tCross = core_time.determine_zero_crossing(
            t=self.time, xTime=self.xTime, typeCross=typeCross
        )
        (
            hWave,
            tWave,
            aCrest,
            aTrough,
            tCrest,
            tTrough,
        ) = core_time.determine_params_individual_waves(
            tCross=tCross, t=self.time, xTime=self.xTime
        )
        return hWave, tWave, aCrest, aTrough, tCrest, tTrough

    def plot(self, savepath: str = "", plot_crossing: bool = False) -> figure.Figure:
        """Plot Series

        Parameters
        ----------
        savepath : str, optional
            path to save figure, by default ""
        plot_crossing : bool, optional
            plot zero crossings in figure, by default False

        Returns
        -------
        figure.Figure
            figure object

        """
        fig = plt.figure()
        plt.plot(self.time, self.xTime, label="series")
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
