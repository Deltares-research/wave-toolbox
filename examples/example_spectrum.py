import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join('..'))

from deltares_wave_toolbox import Spectrum, Series

from deltares_wave_toolbox import create_spectrum_jonswap, compute_spectrum_params, compute_spectrum_freq_serie, spectrum2timeseries, time2freq_nyquist, compute_spectrum_time_serie


## create JONSWAP Spectrum ##
ff = np.linspace(0.01,2,100)

spec = create_spectrum_jonswap(f=ff, fp=0.1, hm0=1)


Hm0 = spec.get_Hm0()
Tps = spec.get_Tps()

spec.plot()

## create Series from spectrum ##
timeseries = spec.create_series(10,1800,0.1)

timeseries.plot()

timeseries.plot()

hExcPerc = timeseries.get_exceedance_waveheight(0.1)
Hs,Ts = timeseries.highest_waves(0.33333)
Hrms  = timeseries.get_Hrms()

spec2 = timeseries.get_spectrum(fres=0.01)

