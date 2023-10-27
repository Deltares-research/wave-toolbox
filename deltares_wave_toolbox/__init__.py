from .series import Series, WaveHeights  # noqa
from .spectrum import Spectrum  # noqa
from .cores import *  # noqa

__version__ = "1.0.0"

# set stylesheet
import matplotlib.pyplot as plt
import os

plt.style.use(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "Style.mplstyle")
)
