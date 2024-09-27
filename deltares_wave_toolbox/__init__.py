from .cores import *  # noqa
from .series import Series, WaveHeights  # noqa
from .spectrum import Spectrum  # noqa

__version__ = "1.0.0"

import os

# set stylesheet
import matplotlib.pyplot as plt

plt.style.use(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "Style.mplstyle")
)
