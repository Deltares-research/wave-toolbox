import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import blackman, hann, boxcar, cosine, hamming, gaussian, welch

sys.path.insert(1, os.getcwd())

import deltares_wave_toolbox as dwt


df = 0.005
f = np.arange(0, 0.5 + df, df)
Tp = 5
fp = 1 / Tp
hm0 = 1.5
gammaPeak = 3.3

dt = 0.5
duration = 3600 * 5

jonswap = dwt.core_wavefunctions.create_spectrum_object_jonswap(f=f, fp=fp, hm0=hm0)

timeseries = jonswap.create_series(0, duration, dt)

plt.figure()
plt.plot(timeseries.time, timeseries.x)

f, Pxx = welch(
    timeseries.x,
    fs=1 / dt,
    window="hann",
    nperseg=256,
    noverlap=None,
    nfft=None,
    detrend="constant",
    return_onesided=True,
    scaling="density",
    axis=-1,
    average="mean",
)


spec = timeseries.get_spectrum(fres=0.01)

plt.figure()
plt.plot(spec.f, spec.S)
plt.plot(f, Pxx)


print("klaar")
