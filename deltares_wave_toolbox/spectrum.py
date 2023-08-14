import numpy as np
import matplotlib.pyplot as plt

from deltares_wave_toolbox.core_spectral import compute_moment, tpd
import deltares_wave_toolbox.core_engine as engine_core




class Spectrum():
    """
    A spectrum class
    """
    # load static methods
    compute_moment = staticmethod(compute_moment)
    tpd = staticmethod(tpd)


    def __init__(self,f,S,D=None):
        f,fSize =engine_core.convert_to_vector(f)
        S,SSize =engine_core.convert_to_vector(S)

        assert engine_core.monotonic_increasing_constant_step(f), 'compute_spectrum_params: Input error: frequency input parameter must be monotonic with constant step size'
        assert fSize[0]==SSize[0], 'compute_spectrum_params: Input error: array sizes differ in dimension'
      
        self.f = f
        self.S = S
        self.D = D

        self.nf = len(f)

        if self.D==None:
            self.spec = '1D'
        else:
            self.spec = '2D'

        
    def __str__(self):
        return f"{self.spec} wave spectrum with {self.nf} number of frequencies"

    def __repr__(self):
        return f"{type(self).__name__} (spec = {self.spec})"
    
    def _set_flim(self,fmin, fmax):
        if ( fmin is None ):
            fmin = self.f[0]
        if ( fmax is None):  
            fmax = self.f[-1]
            return fmin, fmax


    def get_Hm0(self,fmin=None,fmax=None):
        fmin, fmax = self._set_flim(fmin, fmax)
        m0  = self.compute_moment(self.f,self.S, 0,fmin,fmax)
        self.Hm0 = 4 * np.sqrt( m0 )
        return self.Hm0
    
    def get_Tps(self,fmin=None,fmax=None):
        fmin, fmax = self._set_flim(fmin, fmax)
        pass

    def get_Tp(self,fmin=None,fmax=None):
        pass

    def get_Tmm10(self,fmin=None,fmax=None):
        fmin, fmax = self._set_flim(fmin, fmax)
        m_1 = self.compute_moment(self.f,self.S,-1,fmin,fmax)
        m0  = compute_moment(self.f,self.S, 0,fmin,fmax)
        self.Tmm10 = m_1 / m0
        return self.Tmm10

    def get_Tm01(self,fmin=None,fmax=None):
        pass

    def get_Tm02(self,fmin=None,fmax=None):
        pass

    def plot(self,savepath=None,fig=None):

        if fig is None:
            fig = plt.figure()
        plt.plot(self.f,self.S)
        plt.xlabel('f [Hz]')
        plt.ylabel('S [m^2/Hz]')
        if savepath is not None:
            plt.savefig(savepath)


