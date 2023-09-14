import numpy as np
import matplotlib.pyplot as plt

from deltares_wave_toolbox.cores.core_spectral import spectrum2timeseries
from deltares_wave_toolbox.cores.core_wavefunctions import compute_moment, tpd, compute_tps
from deltares_wave_toolbox.series import Series
import deltares_wave_toolbox.cores.core_engine as engine_core




class Spectrum():
    """
    The wave Spectrum class
    """
    # load static methods 
    compute_moment = staticmethod(compute_moment)
    tpd = staticmethod(tpd)
    spectrum2timeseries = staticmethod(spectrum2timeseries)
    compute_tps = staticmethod(compute_tps)


    def __init__(self,f,S,D=None):
        """_The innit function

        Args:
            f (array): array with frequencies
            S (array): array with energy density
            D (array, optional): array with directions for 2D spectrum. Defaults to None.
        """
        f,fSize =engine_core.convert_to_vector(f)
        S,SSize =engine_core.convert_to_vector(S)

        assert engine_core.monotonic_increasing_constant_step(f), 'compute_spectrum_params: Input error: frequency input parameter must be monotonic with constant step size'
        assert fSize[0]==SSize[0], 'compute_spectrum_params: Input error: array sizes differ in dimension'
      
        self.f = f
        self.S = S
        self.D = D
        self.nf = len(f)

        ## 1D or 2D spectrum
        if self.D==None:
            self.spec = '1D'
        else:
            self.spec = '2D'

        
    def __str__(self):
        return f"{self.spec} wave spectrum with {self.nf} number of frequencies"

    def __repr__(self):
        return f"{type(self).__name__} (spec = {self.spec})"
    
    def _set_flim(self,fmin, fmax):
        """Set frequency limits

        Args:
            fmin (float): Minimum frequency
            fmax (float): Maximum frequency

        Returns:
            float: minimum and maximum frequency
        """
        if ( fmin is None ):
            fmin = self.f[0]
        if ( fmax is None):  
            fmax = self.f[-1]
            return fmin, fmax


    def get_Hm0(self,fmin=None,fmax=None):
        """ Compute Hm0 of spectrum

        Args:
            fmin (float, optional): Minimum frequency. Defaults to None.
            fmax (float, optional): Maximum frequency. Defaults to None.

        Returns:
            float: Hm0
        """
        fmin, fmax = self._set_flim(fmin, fmax)
        m0  = self.compute_moment(self.f,self.S, 0,fmin,fmax)
        self.Hm0 = 4 * np.sqrt( m0 )
        return self.Hm0
    
    def get_Tps(self,fmin=None,fmax=None):
        """ Compute Tps (smoothed peak period) of spectrum

        Args:
            fmin (float, optional): Minimum frequency. Defaults to None.
            fmax (float, optional): Maximum frequency. Defaults to None.

        Returns:
            float: Tps
        """
        fmin, fmax = self._set_flim(fmin, fmax)
        iFmin = np.where(self.f>=fmin)[0][0]  
        iFmax = np.where(self.f<=fmax)[0][-1]  
        fMiMa = self.f[iFmin:iFmax+1]
        SMiMa = self.S[iFmin:iFmax+1]
        Tps = compute_tps(fMiMa, SMiMa)
        return Tps

    def get_Tp(self,fmin=None,fmax=None):
        """ Compute Tp (peak period) of spectrum

        Args:
            fmin (float, optional): Minimum frequency. Defaults to None.
            fmax (float, optional): Maximum frequency. Defaults to None.

        Returns:
            float: Tp
        """
        # --- Make separate arrays containing only part corresponding to
        #     frequencies between fmin and fmax 
        #     Note: first zero selects first element of tuple.
        fmin, fmax = self._set_flim(fmin, fmax)
        iFmin = np.where(self.f>=fmin)[0][0] 
        iFmax = np.where(self.f<=fmax)[0][-1]  
        fMiMa = self.f[iFmin:iFmax+1]
        SMiMa = self.S[iFmin:iFmax+1]
        # --- Compute peak period -----------------------------------------------
        Smax = max( SMiMa )
        imax = np.where(SMiMa == Smax)[0] # matlab find( SMiMa == Smax );
        imax = imax.astype(int)
        ifp  = max(imax)
        fp   = fMiMa[ifp]
        #
        if np.all(ifp==None):  # matlab isempty(ifp)
            ifp = 1
            fp  = fMiMa[ifp]
        self.Tp = 1 / fp
        return self.Tp

    def get_Tmm10(self,fmin=None,fmax=None):
        """ Compute Tmm10 spectral period) of spectrum

        Args:
            fmin (float, optional): Minimum frequency. Defaults to None.
            fmax (float, optional): Maximum frequency. Defaults to None.

        Returns:
            float: Tmm10
        """
        fmin, fmax = self._set_flim(fmin, fmax)
        m_1 = self.compute_moment(self.f,self.S,-1,fmin,fmax)
        m0  = compute_moment(self.f,self.S, 0,fmin,fmax)
        self.Tmm10 = m_1 / m0
        return self.Tmm10

    def get_Tm01(self,fmin=None,fmax=None):
        """ Compute Tm01 of spectrum

        Args:
            fmin (float, optional): Minimum frequency. Defaults to None.
            fmax (float, optional): Maximum frequency. Defaults to None.

        Returns:
            float: Tm01
        """
        fmin, fmax = self._set_flim(fmin, fmax)
        m0  = compute_moment(self.f,self.S, 0,fmin,fmax)
        m1  = compute_moment(self.f,self.S, 1,fmin,fmax)
        self.Tm01  = m0 / m1
        return self.Tm01

    def get_Tm02(self,fmin=None,fmax=None):
        """ Compute Tm02 of spectrum

        Args:
            fmin (float, optional): Minimum frequency. Defaults to None.
            fmax (float, optional): Maximum frequency. Defaults to None.

        Returns:
            float: Tm02
        """
        fmin, fmax = self._set_flim(fmin, fmax)
        m0  = compute_moment(self.f,self.S, 0,fmin,fmax)
        m2  = compute_moment(self.f,self.S, 2,fmin,fmax)
        self.Tm02  = np.sqrt( m0 / m2 )
        return self.Tm02 

    def create_series(self, tstart, tend, dt):
        """ Construct series from Spectrum with random phase

        Args:
            tstart (float): Start time of time series
            tend (float): End time of time series
            dt (float): Time step

        Returns:
            object: Series class with time series
        """
        [t,xTime] = spectrum2timeseries(self.f,self.S,tstart,tend,dt)
        return Series(t, xTime)

    def plot(self,savepath=None,fig=None):
        """ Plot spectrum

        Args:
            savepath (str, optional): path to save figure. Defaults to None.
            fig (figure object, optional): figure object. Defaults to None.
        """

        if fig is None:
            fig = plt.figure()
        plt.plot(self.f,self.S)
        plt.grid('on')
        plt.xlabel('f [$Hz$]')
        plt.ylabel('S [$m^2/Hz$]')
        if savepath is not None:
            plt.savefig(savepath)


