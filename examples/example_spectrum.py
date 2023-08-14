import sys
import os
import numpy as np

sys.path.append(os.path.join('..'))

from deltares_wave_toolbox import Spectrum, create_spectrum_jonswap



f= np.linspace(0.01,2,100)

S = create_spectrum_jonswap(f=f,fp=0.1,hm0=1)    



spec = Spectrum(f,S)

Hm0 = spec.get_Hm0()

spec.plot()


print()