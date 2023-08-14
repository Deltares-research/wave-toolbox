"""
Spyder Editor

# Copyright notice
#     Copyright (c) 2008  Deltares.
#       Ivo Wenneker
#
#       ivo.wenneker@deltares.nl
#
#       Rotterdamseweg 185
#       Delft
#
#     All rights reserved.
#     This routine is part of the WaveLab software system.
#     Using this routine one is submitted to the license
#     agreement for the WaveLab system.
#     Permission to copy or distribute this software or documentation
#     in hard or soft copy granted only under conditions as described
#     by the license obtained from DELTARES.
#     No part of this publication may be reproduced, stored in a
#     retrieval system or be transmitted by any means, electronic,
#     mechanical, photocopy, recording, or otherwise, without written
#     permission from DELTARES.

# Version <http://svnbook.red-bean.com/en/1.5/svn.advanced.props.special.keywords.html>
# Created: 
# Created 

# $Id:  $
# $Date:  $
# $Author:  $
# $Revision:  $
# $HeadURL:  $
# $Keywords: $
"""
# --- python modules 
from enum import Enum
import math
import scipy.integrate as integrate
from scipy.fftpack import fft, ifft
import numpy as np
import copy
 

# --- toolbox modules 
import deltares_wave_toolbox.core_engine as engine_core


class DataWindowTypes(Enum):
    UNKNOWN =-1
    HANNING =0
    RECTANGULAR =1

def tpd(freqs:np.ndarray=None,spectrum:np.ndarray=None)->float:
    '''
    TpD : Function which calculates the spectral period (s)
    
    Input (key,value) :
    'Spectrum',value   : numeric, array of variance density values (m2/Hz)  
    'Frequencies',value: numeric, array of frequencies (Hz) 
     Output :
     out                : numeric, spectral period (s)
             
    Note: For definition of TpD: Overstap van piekperiode naar spectrale periode bij ontwerp van steenzettingen
    '''
       
    freqs,f_size =engine_core.convert_to_vector(freqs)
    spectrum,spectrum_size =engine_core.convert_to_vector(spectrum)
 
       
            
    # --- calculate the spectral period (TPD) (s)
    max_spectum = max(spectrum)*0.8
    itemp = np.where(spectrum / max_spectum >=0.8 )[0]  #temp=freqs( (spectrum ./ maxSpectum) >= 0.8);
    temp = freqs[itemp]
    fp_limits = [min(temp), max(temp)] #clear temp
            
    #  --- compute zeroth and first moment for selected frequency interval.
    m0 = compute_moment(freqs,spectrum,0,fp_limits[0],fp_limits[1])
    m1 = compute_moment(freqs,spectrum,1,fp_limits[0],fp_limits[1]);
            
    # --- calculate TpD based on spectral moments.
    return m0/m1; 
    


def frequency_averaging(f=None,sFreq=None,dfDesired=None):
    '''
    FREQUENCYAVERAGING  Band averaging of given variance density spectrum 

    This function performs a band averaging on a given variance density 
    spectrum sFreq = sFreq(f) on a frequency axis f onto a coarser 
    frequency axis with frequency spacing dfDesired.


    Parameters
    ----------
    f     :  
          frequency axis [Hz]
    sFreq :     
          variance density spectrum as function of frequency f
    dfDesired :
        (optional parameter) desired frequency spacing in Hertz on 
        which sFreq must be computed.
        If this parameter is omitted, then dfDesired = f(1) - f(0)

    Returns
    -------
    fCoarse  :
             frequency axis of band averaged spectrum. The frequency
            spacing is (close to) dfDesired
    sFreqCoarse :
            band averaged variance density spectrum

    Remark:
        for tranlation between matlab and python:
        see https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
   
    Syntax:
        fCoarse,sFreq = frequency_averaging(f,sFreq,dfDesired)

    Example
    ------- 
    >>> import numpy as np
    >>> df        = 0.01 #Hz
    >>> f         = np.arange(0,100)*df
    >>> sFreq     = np.random.normal(loc=1, scale=2, size=len(f))
    >>> dfDesired =0.02
    >>> fCoarse,sFreq = frequency_averaging(f,sFreq,dfDesired)

    '''
    
    
        # convert input to array type to be able to handle input like e.g. f = [0.2,0.4]
    f,fSize =engine_core.convert_to_vector(f)
    sFreq,sFreqSize =engine_core.convert_to_vector(sFreq)
         
    if (fSize[1] >1 or sFreqSize[1]>1):
        raise ValueError('frequency_averaging: Input error: input should be 1d arrays')
    
    # --- Determine some coefficients
    nFine   = fSize[0]
    nFactor = 1
    if (not (dfDesired is None)):
      nFactor = round( dfDesired / (f[1] - f[0]) )
    
    # --- Avoid nFactor being equal to zero, which may occur if 
    #     dfDesired < 0.5 * (f2 - f1)
    nFactor = max( nFactor,1 )
    nCoarse = math.floor( nFine / nFactor )

    # --- Initialize arrays
    fCoarse     = np.zeros( len( f[0:nCoarse]))
    sFreqCoarse = np.zeros( len( fCoarse ),dtype=engine_core.get_parameter_type(sFreq[0])) # determine parameter type complex or float?

    # --- Perform the averaging
    for iFreq in np.arange(0,nCoarse):   # before np.arange(0,nCoarse) 
       # note: python arrays start at index zero!!
       ilow               = int((iFreq) * nFactor ) 
       ihigh              = int((iFreq+1) * nFactor )
    
       # note: for example f[1:1] is empty where as f[1:2] is equal to f[1]
       fCoarse[iFreq]     = np.mean( f[ilow:ihigh] )
       sFreqCoarse[iFreq] = np.mean( sFreq[ilow:ihigh] )

    return fCoarse,sFreqCoarse

def unfold_spectrum(f,xFreq,isOdd):
    '''
    UNFOLDSPECTRUM  Unfolds a folded discrete Fourier transform 

    This function unfolds a folded discrete Fourier transform xFreq that 
    is given at frequency axis f. Note that this frequency axis goes up to 
    the Nyquist frequency. Parameter isOdd indicates whether the underlying 
    original time signal - of which xFreq is the discrete Fourier transform
    - has even (isOdd=0) or odd (isOdd=1) time points.

    The unfolded discrete Fourier xFreqTot, at frequency axis fTot, can be
    inverted back to time domain using function freq2time
 

    Parameters
    ----------
    f     :   
           1D real array containing frequency values, for folded Fourier 
           transform. The frequency axis runs from 0 to the Nyquist
           frequency. 
    xFreq :
          1D array (complex!) containing the folded Fourier coefficients 
          of original time series. The value xFreq(i) must be the Fourier
          coefficient at frequency f(i). The number of elements in f and
          xFreq are the same. 
    isOdd :
          logical indicating whether nT, the number of time points in 
          original time series, is even (isOdd=0) or odd (isOdd=1)


    Returns
    -------
    fTot     :
             1D real array containing frequency values, for unfolded
             Fourier transform. The frequency axis runs from 0 to twice
             the Nyquist frequency. Array f contains as many elements as
             the original time series. 
    xFreqTot :
             1D array (complex!) containing the unfolded Fourier
             coefficients of original time series. The value xFreqTot(i)
             must be the Fourier coefficient at frequency fTot(i). The
             number of elements in fTot and xFreqTot are the same. 

    Remark:
        See also time2freq_nyquist, freq2time
        see https://numpy.org/doc/stable/user/numpy-for-matlab-users.html

    Syntax:
        fTot,xFreqTot = unfold_spectrum(f,xFreq,isOdd)
   
    Example
    -------
    >>> import numpy as np
    >>> dt            = 0.1 # s
    >>> t             = np.arange(0,100) *dt  # Time axis
    >>> a1 = 0.5; w1  = 2*np.pi/5; phi1 = 0.35  # Wave component 1 
    >>> a2 = 0.7; w2  = 2*np.pi/6; phi2 = 0.96  # Wave component 2 
    >>> y = a1*np.cos(w1*t+phi1) + a2*np.cos(w2*t+phi2)
    >>> # --- Compute folded discrete Fourier transform
    >>> f,yFreq,isOdd = time2freq_nyquist(t,y); 
    >>> # --- Unfold the discrete Fourier transform 
    >>> fTot,yFreqTot = unfold_spectrum(f,yFreq,isOdd) 
    >>> # --- Return to time domain
    >>> yTime         = freq2time(yFreqTot)  # yTime must be identical to y
   
    '''
    
    # convert input to array type to be able to handle input like e.g. f = [0.2,0.4]
    f,fSize =engine_core.convert_to_vector(f)
    xFreq,xFreqSize =engine_core.convert_to_vector(xFreq)
         
    if (fSize[1] >1 or xFreqSize[1]>1):
        raise ValueError('unfold_spectrum: Input error: input should be 1d arrays')

    nF    = fSize[0]
    nFTot = ( nF - 1 )*2 + isOdd

    # --- Construct frequency axis
    # Note that the first half part of the frequency axis is equal to the input
    # frequency axis as given in freq
    df = f[1] - f[0]
    fTot = np.arange(0,nFTot)*df    #fTot = [0:nFTot-1]*df
    
    xFreqTot         = np.zeros( (len(fTot)), dtype=complex )
    xFreqTot[0:nF]   = copy.deepcopy(xFreq[0:nF]) #.reshape(nF,1) #xFreq[0:nF]
    
    # Arrays f and xFreqTot are column vectors. Apply a flip upside-down
    xFreqTot[(nF-1)+isOdd:nFTot] = np.flipud( np.conj( xFreqTot[1:nF] ) )
     
    # not     
    #if (  dims[1] == nF ):
    #   # Arrays f and xFreqTot are column vectors. Apply a flip upside-down
    #   xFreqTot[(nF-1)+isOdd:nFTot] = np.flipud( np.conj( xFreqTot[1:nF] ) )
    #else:
    #   # Arrays f and xFreqTot are row vectors. Apply a flip left-right
    #   xFreqTot[(nF)+isOdd:nFTot] = np.fliplr( np.conj( xFreqTot[1:nF] ) )
    
    return fTot,xFreqTot

def coherence(f,xFreq1,xFreq2,dfDesired):
    '''
    COHERENCE  Function to compute the coherence in spectral domain.

    This function computes the coherence (magnitude-squared cohorence) of
    two complex spectral signal xFreq1 = xFreq1(f) and xFreq2 = xFreq2(f),
    given on frequency axis f. The output coh2 = coh2(f_coh2) is given on a
    frequency axis with frequency resolution dfDesired. 
    Note: the coherence is real, and always between 0 and 1.

    Parameters
    ----------
     f        :
              1d array frequency axis [Hz]
    xFreq1    : 
              wave spectrum 1 of complex Fourier coefficients
    xFreq2    :
              wave spectrum 1 of complex Fourier coefficients
    dfDesired :
              (optional parameter) desired frequency spacing in Hertz on 
              which sFreq must be computed.
              If this parameter is omitted, then dfDesired = f(2) - f(1)

    Returns
    -------
    f_coh2   :
             frequency axis of coherence. The frequency
             spacing is (close to) dfDesired
    coh2     :
             coherence (magnitude-squared coherence). 

    Remark:     
         for tranlation between matlab and python:
         see https://numpy.org/doc/stable/user/numpy-for-matlab-users.html

    Syntax:
         f_coh2,coh2 = coherence(f,xFreq1,xFreq2,dfDesired)

    Example  
    -------
    >>> import numpy as np
    >>> 
    >>> 
    >>> 
    >>> 
    
    '''
    
    
    
    
    #
    # --- Compute auto-spectral and cross-spectral (absolute value) densities 
    #     on original frequency axis
    #     Note: Normalisation factors (2 / (df * Ntime * Ntime)) may be omitted,
    #     since they cancel out
    S11 = abs( xFreq1*np.conj(xFreq1) )
    S22 = abs( xFreq2*np.conj(xFreq2) )
    S12 = abs( xFreq1*np.conj(xFreq2) )

    # --- Apply band averaging
    f_coh2,S11_b = frequency_averaging(f,S11,dfDesired)
    f_coh2,S22_b = frequency_averaging(f,S22,dfDesired)
    f_coh2,S12_b = frequency_averaging(f,S12,dfDesired)

    # --- Omit division by zero, by putting the zeros (very small numbers) to a small
    #     number.
    small_number    = 1E-10;

    ismall          = np.where( S11_b < small_number )[0]
    S11_b[ismall ] = small_number

    ismall          = np.where( S22_b < small_number )[0]
    S22_b[ ismall ] = small_number;

    # --- Compute coherence
    coh2 = ( S12_b )**2 / ( S11_b * S22_b )

    return f_coh2,coh2

def freq2time(xFreq):
    '''
    FREQ2TIME  Transforms (unfolded) discrete Fourier transform back to time 
                signal

    This function transforms a given discrete and unfolded Fourier 
    transform xFreq (in general a complex quantity!) back to time domain. 
    Note that the input Fourier transform xFreq must be unfolded. A given
    folded Fourier transform can be unfolded using the function
    unfoldspectrum. 


    Parameters
    ----------
    xFreq :
          1D array (complex!) containing unfolded Fourier coefficients.

    Returns
    -------
    xTime :
          1D real array containing time series of the signal. 

    Remark:     
         for tranlation between matlab and python:
         see https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
         
         See also time2freq, time2freqnyquist, unfoldspectrum

    Syntax:
         xTime = freq2time(xFreq)
         
    Example  
    -------
    >>> import numpy as np
    >>> # --- Create time signal
    >>> dt =0.1   # s
    >>> t = np.arange(0,100)*dt  # Time axis 
    >>> a1 = 0.5; w1 = 2*np.pi/5; phi1 = 0.35  # Wave component 1 
    >>> a2 = 0.7; w2 = 2*np.pi/6; phi2 = 0.96  # Wave component 2 
    >>> y = a1*np.cos(w1*t+phi1) + a2*np.cos(w2*t+phi2)
    >>> # --- Compute discrete Fourier transform 
    >>> f,yFreq = time2freq(t,y)
    >>> # --- Return to time domain 
    >>> yTime = freq2time(yFreq)  # yTime must be identical to y 
    
    '''
  
    xFreq,xFreqSize =engine_core.convert_to_vector(xFreq)
    
    
    #
    # Check on input arguments
    #  xFreq must be a 1D array
    # assert 
    
    if ( xFreqSize[1]>1):
        raise ValueError('freq2time: Input error: input should be 1d arrays')
      
    # Computational core
    nF    = xFreqSize[0]
    xTime = np.real(np.fft.ifft(xFreq,nF) )
    
    return xTime

def time2freq(t,xTime):
    '''
    TIME2FREQ  Computes the discrete Fourier transform coefficients (on 
               unfolded frequency axis) of a given of time signal
 
    This function computes the Fourier coefficients xFreq (in general
    complex quantities!) on frequency axis f (hence, xFreq = xFreq(f)),
    from a given time signal Xtime on time axis t (hence, xTime =
    xTime(t)). 
 
    The Fourier transform is not folded. This means, that the number of
    elements in arrays f and xFreq is identical to nT (nT: the number of
    elements in arrays t and xTime). Note that the Fourier coefficients in
    xFreq have a complex conjugate symmetry around the Nyquist frequency.  
    Transforming the signal xFreq = xFreq(f) back to time domain can be
    done with the function freq2time

    Parameters
    ----------
    t      :
           1D real array containing time values. The numbers in the 
           array t must be increasing and uniformly spaced (uniform time
           step). The initial time t(1) can be any value (so it is not
           obligatory to have t(1) = 0)
    xTime  :
           1D real array containing signal values, i.e. the time
           series of the signal. The value xTime(i) must be the signal 
           value at time t(i). The number of elements in t and xTime must
           be the same

    Returns
    -------
    f     :
          1D real array containing frequency values, for unfolded Fourier
          transform. The frequency axis runs from 0 to twice the Nyquist 
          frequency. Array f contains as many elements as array xTime. 
    xFreq :
          1D array (complex!) containing the unfolded Fourier
          coefficients of xTime. The value xFreq(i) must be the Fourier 
          coefficient at frequency f(i). The number of elements in f and 
          xFreq are the same. This number is the same as the number of 
          elements in t and xTime.
    
    Remark:     
         for tranlation between matlab and python:
         see https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
         
    Syntax:
         f,xFreq = time2freq(t,xTime)

    See also time2freq, time2freq_nyquist, unfold_spectrum
                  
    Example  
    -------
    >>> import numpy as np
    >>> # --- Create time signal
    >>> dt =0.1 # s
    >>> t = np.arange(0,100)*dt  # Time axis 
    >>> a1 = 0.5; w1 = 2*np.pi/5; phi1 = 0.35  # Wave component 1 
    >>> a2 = 0.7; w2 = 2*np.pi/6; phi2 = 0.96  # Wave component 2 
    >>> y = a1*np.cos(w1*t+phi1) + a2*np.cos(w2*t+phi2)
    >>> # --- Compute discrete Fourier transform 
    >>> f,yFreq = time2freq(t,y)
    
    '''
    # Check on input
    # aktie Ivo: toevoegen de volgende checks: 
    # t is stijgend, met constante dt
    # length(t) = length(xtime)
    # t en xTime zijn 1D arrays
    t,tSize =engine_core.convert_to_vector(t)
    xTime,xTimeSize =engine_core.convert_to_vector(xTime)
    
    if (tSize[1] >1 or xTimeSize[1]>1):
        raise ValueError('time2freq: Input error: input should be 1d arrays')
      
    if (not engine_core.monotonic_increasing_constant_step(t)):
        raise ValueError('time2freq: Input error: time input parameter must be monotonic with constant step size')
      
    if (not (tSize[0]==xTimeSize[0])):
        raise ValueError('time2freq: Input error: array sizes differ in dimension')


    # Computational core
    # --- Initialize some constants
    #     Note: T = nT*dt = t(end) - t(1) + dt
    nT = tSize[0];
    dt = t[1] - t[0];
    T  = nT*dt

    # --- Create frequency axis
    df = 1.0 / T;
    #f  =np.arange(0,nT)*df
    f  =np.arange(0,nT)*df # exlcude f=0
    
    #f  = f.reshape(len(f),1)
    #xTime = xTime.reshape(len(f),1)
    
#    if ( len(t) == nT ):
#       # Array t is column vector. Array f must be transposed to become column
#       # vector as well
#       # f = f'  (matlab)
#       f = f.T
 
    # --- Compute Fouriertransform
    xFreq= np.fft.fft( xTime )
 
    return f,xFreq
 
def time2freq_nyquist(t,xTime):
    '''
    TIME2FREQNYQUIST  Computes the discrete Fourier transform coefficients (on 
                  folded frequency axis) of a given of time signal

    This function computes the Fourier coefficients xFreq (in general complex
    quantities!) on frequency axis f (hence, xFreq = xFreq(f)), from a given 
    time signal Xtime on time axis t (hence, xTime = xTime(t)). 

    The Fourier transform is folded. This means, that the number of elements
    in arrays f and xFreq is identical to floor(nT/2) + 1 (nT: the 
    number of elements in arrays t and xTime). This means that the 
    the Fourier coefficients are computed up to the Nyquist frequency.
    Transforming the signal xFreq = xFreq(f) back to time domain can be done 
    using first the function unfoldspectrum and after that the function 
    freq2time. Parameter isOdd=1 if nT is odd, and isOdd=0 if nT is even. 


    Parameters
    ----------
    
    Input:
    t     :  
          1D real array containing time values. The numbers in the 
          array t must be increasing and uniformly spaced (uniform 
          time step)
    xTime :
          1D real array containing signal values, i.e. the time
          series of the signal. The value xTime(i) must be the signal 
          value at time t(i). 
          The number of elements in t and xTime must be the same
    
    
    Returns
    -------
    f     :
          1D real array containing frequency values, for folded Fourier 
          transform. The frequency axis runs from 0 to the Nyquist
          frequency. The number of elements in array f is close to half
          the number of elements in array xTime. To be precise, length(f)
          = floor(nT/2) + 1, with nT the number of elements in array
          xTime
    xFreq :
          1D array (complex!) containing the folded Fourier coefficients 
          of xTime. The value xFreq(i) must be the Fourier coefficient
          at frequency f(i). The number of elements in f and xFreq are 
          the same. 
    isOdd : 
          logical indicating whether nT, the number of time points in 
          xTime, is even (isOdd=0) or odd (isOdd=1)
    
    Remark:
        for tranlation between matlab and python:
        see https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
   
    Syntax:
        f,xFreq,isOdd = time2freq_nyquist(t,xTime)
        
    See also time2freq, unfold_spectrum, freq2time

    Example
    ------- 
    >>> import numpy as np
    >>> dt        = 0.1 #Hz
    >>> t         = np.arange(0,100)*dt
    >>> a1 = 0.5; w1 = 2*np.pi/5; phi1 = 0.35;  # Wave component 1
    >>> a2 = 0.7; w2 = 2*np.pi/6; phi2 = 0.96;  # Wave component 2
    >>> y = a1*np.cos(w1*t+phi1) + a2*np.cos(w2*t+phi2)
    >>> print(t)
    >>> print(y)
    >>> # --- Compute discrete Fourier transform
    >>> f,yFreq,isOdd = time2freq_nyquist(t,y);
 
    ''' 
     # --- Ensure array input is of type ndarray.
    
      
    ## Check on input
    # Aktie Ivo: toevoegen de volgende checks: 
    # t is stijgend, met constante dt
    # length(t) = length(xtime)
    # t en xTime zijn 1D arrays

    t,tSize =engine_core.convert_to_vector(t)
    xTime,xTimeSize =engine_core.convert_to_vector(xTime)
    
    if (tSize[1] >1 or xTimeSize[1]>1):
        raise ValueError('time2freq_nyquist: Input error: input should be 1d arrays')
      
    if (not engine_core.monotonic_increasing_constant_step(t)):
        raise ValueError('time2freq_nyquist: Input error: time input parameter must be monotonic with constant step size')
      
    if (not (tSize[0]==xTimeSize[0])):
        raise ValueError('time2freq_nyquist: Input error: array sizes differ in dimension')

    ## Computational core Transform time signal to frequency domain, over 
    #  frequency axis up to twice the Nyquist frequency
    fTotal, xFreqTotal = time2freq(t,xTime)
   
    # --- Number of time points, and check whether this number is even or odd
    nT    = tSize[0]
    isOdd = nT % 2 

    # --- Index in array that corresponds to the Nyquist frequency
    #     Nyquist frequency = 1 / (2*dt)
    iNyq = math.floor( nT/2 ) + 1

    # --- Take part of the signal up to the Nyquist frequency
    f     = fTotal[ 0:iNyq ]
    xFreq = xFreqTotal[ 0:iNyq ]
   
    #
    return f, xFreq, isOdd

def compute_spectrum_params(f=None,S=None,fmin=None,fmax=None):
    """
    COMPUTE_SPECTRUM_PARAMS  Computes spectral parameters of given spectrum

    This function computes several spectral wave parameters of a given 1D
    spectrum


    Parameters
    ----------
    f    : array double (1D)
         1D array representing frequency axis (unit: Hz)
    S    : array double (1D) 
         1D array representing variance density spectrum (units: m2/Hz).
    fmin : double
         (optional argument) lower bound of the moment integral (unit: Hz)
    fmax : double 
         (optional argument) upper bound of the moment integral (unit: Hz)

    Returns
    -------
    Hm0   : double
          wave height (units: m)
    Tp    : double 
          peak period (units: s)
    Tps   : double
          smoothed peak period (units: s)
    Tmm10 : double
          wave period based on (-1) and (0) moments (units: s)
    Tm01  : double
          wave period based on (0) and (1) moments (units: s)
    Tm02  : double
          wave period based on (0) and (2) moments (units: s)
    

    Syntax:
          [Hm0,Tp,Tps,Tmm10,Tm01,Tm02] = compute_spectrum_params(f,S,fmin,fmax)


    Example 
    -------
    >>> import numpy as np
    >>> f = np.arange(0,1,0.1)  
    >>> Tp = 5.0
    >>> Hm0 = 1.0
    >>> fmin =0.01
    >>> fmax =1.0
    >>> sPM = create_spectrum_piersonmoskowitz(f,1/Tp,Hm0)
    >>> [Hm0,Tp,Tps,Tmm10,Tm01,Tm02] = compute_spectrum_params(f,sPM,fmin,fmax)
 
    See also computemoment
    
    """ 
    
    # --- Ensure array input is of type ndarray.
    f,fSize =engine_core.convert_to_vector(f)
    S,SSize =engine_core.convert_to_vector(S)
    
    if (fSize[1] >1 or SSize[1]>1):
        raise ValueError('compute_spectrum_params: Input error: input should be 1d arrays')
      
    if (not engine_core.monotonic_increasing_constant_step(f)):
        raise ValueError('compute_spectrum_params: Input error: frequency input parameter must be monotonic with constant step size')
      
    if (not (fSize[0]==SSize[0])):
        raise ValueError('compute_spectrum_params: Input error: array sizes differ in dimension')
    
    #
    # --- Find values of fmin and fmax
    if ( fmin is None ):
       fmin = f[0]
    if ( fmax is None):  
       fmax = f[fSize[0]-1]

    # --- Compute moments
    m_1 = compute_moment(f,S,-1,fmin,fmax)
    m0  = compute_moment(f,S, 0,fmin,fmax)
    m1  = compute_moment(f,S, 1,fmin,fmax)
    m2  = compute_moment(f,S, 2,fmin,fmax)

    # --- Compute wave height -----------------------------------------------
    Hm0 = 4 * np.sqrt( m0 )

    # --- Put values to -999 (exception value) in situation that wave height is 
    #     (virtually) zero
    if ( Hm0 < 1e-6 or np.isnan(Hm0) ):
       Hm0   = np.nan
       Tp    = np.nan
       Tps   = np.nan
       Tmm10 = np.nan
       Tm01  = np.nan
       Tm02  = np.nan
       return  [Hm0,Tp,Tps,Tmm10,Tm01,Tm02]

    # --- Compute mean wave periods -----------------------------------------
    Tmm10 = m_1 / m0
    Tm01  = m0 / m1
    Tm02  = np.sqrt( m0 / m2 )

    # --- Make separate arrays containing only part corresponding to
    #     frequencies between fmin and fmax 
    #     Note: first zero selects first element of tuple.
    iFmin = np.where(f>=fmin)[0][0]  # matlab find( f>= fmin,1,'first');
    iFmax = np.where(f<=fmax)[0][-1]  # matlab find( f<= fmax,1,'last');
    fMiMa = f[iFmin:iFmax+1]
    SMiMa = S[iFmin:iFmax+1]

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
    Tp = 1 / fp

    # --- Compute smoothed peak period --------------------------------------
    Tps = compute_tps( fMiMa,SMiMa )

    return [Hm0,Tp,Tps,Tmm10,Tm01,Tm02]

def compute_moment(f=None,S=None,m=None,fmin=None,fmax=None):
    """
    COMPUTE_MOMENT  Computes the spectral moment

    This function computes the m'th order spectral moment
    of variance density spectrum S=S(f), with f the frequency axis, 
    over frequency domain [fmin,fmax].
 
    It is required that fmin >= f_in(1). 
 
    It is not required to have fmax <= f_in(end). So it ok to have fmax =
    Inf.
    If fmax>f(end), then the moment consists of the summation of two parts:
    (1) Integration of (f_in^m * S), with given S, over [fmin,f(end)]
    (2) Exact integration of (f^m * S_lim) over [f(end),fmax], where S_lim
        is a high-frequency f^(-5) tail.
    Typically, in such cases one puts fmax = Inf.
 

    Parameters
    ----------
    f     : array double (1D)
          1D array representing frequency axis (unit: Hz)
    S     : array double (1D)
          1D array representing variance density spectrum (units: m2/Hz).
    m     : integer 
          order of moment (integer value)
    fmin  : double 
          (optional argument) lower bound of the moment integral (unit: Hz)
    fmax  : double
          (optional argument) upper bound of the moment integral (unit: Hz)

    Returns
    -------
    moment : double  
           the computed moment
    
    Syntax:
       moment = compute_moment(f,S,m,fmin,fmax)

    Example 
    -------
    >>> import numpy as np
    >>> f = np.arange(0,1,0.1)  
    >>> Tp = 5.0
    >>> Hm0 = 1.0
    >>> fmin =0.01
    >>> fmax =1
    >>> sPM = create_spectrum_piersonmoskowitz(f,1/Tp,Hm0)
    >>> m_1 = compute_moment(f,sPM,-1)
    >>> m0  = compute_moment(f,sPM,0)
    >>> m1  = compute_moment(f,sPM,1)
    >>> m2  = compute_moment(f,sPM,2)

    See also wavelab function computemoment, integral1d
    """
    
    # --- Ensure array input is of type ndarray.
    f,fSize =engine_core.convert_to_vector(f)
    S,SSize =engine_core.convert_to_vector(S)
    
    if (fSize[1] >1 or SSize[1]>1):
        raise ValueError('compute_moment: Input error: input should be 1d arrays')
      
    if (not engine_core.monotonic_increasing_constant_step(f)):
        raise ValueError('compute_moment: Input error: frequency input parameter must be monotonic with constant step size')
    
        
    if (not (fSize[0]==SSize[0])):
        raise ValueError('compute_moment: Input error: array sizes differ in dimension')
    
    # --- Make sure that f and S are either both column vectors or both row vectors
    #if ( size(f_in,1) == size(S,1) ):
    #    f = f_in
    #else
    #    f = f_in'

    # --- Remove the possible situation that f=0 in combination with m<0. This
    #     would lead to division by zero
    if ( m < 0 and f[0] == 0 ):
       freq = f[1:]   # matlab  f(2:end);
       spec = S[1:]   # matlab  S(2:end);
    else:
       freq = f
       spec = S
    
           
    # --- Compute the integrand, that is the product of f^m * S  (using lambda is an alternatief for using def <function name> :)
    func_integrand = lambda freq,m,spec:freq**(m) * spec    # matlab freq.^(m) .* spec;
    integrand = func_integrand(freq, m, spec)
    
    
    # --- Depending on number of input arguments, compute the moment integral
    if ( fmin is None or fmax is None ): # integrate over all values in freq interval.
        
        moment = integrate.simps(integrand,freq)     # moment = integral1d(freq,integrand) 
    else:  # integrate over all values in sub interval.
        # fmin and fmax are given
        fminn = fmin
        if ( m < 0 and f[0] == 0 ):    # matlab f(1)
           if ( fmin == 0 ):
              fminn = f[1]   # matlab f(2)
        
        #
        if ( fmax <= freq[len(freq)-1]):   # matlab freq(end))      NOTE: USE len(freq) here instead of Nf because length is altered on lines 741                                        
           ifminn = engine_core.approx_array_index(freq, fminn)
           ifmax  = engine_core.approx_array_index(freq, fmax) +1   # due to range specification ifminn:ifmax. e.g. S[0:Nf] runs from S[0] to S[Nf-1] (S[Nf-1] has a value and not S[Nf])
           moment = integrate.simps(integrand[ifminn:ifmax],freq[ifminn:ifmax])     # moment = integral1d(freq,integrand,fminn,fmax);
            
        else:
           # 1: Integral over [fminn,freq(end)]
           ifminn = engine_core.approx_array_index(freq, fminn)
           ifmax  = engine_core.approx_array_index(freq, freq[len(freq)-1]) +1  # due to range specification ifminn:ifmax.  e.g. S[0:Nf] runs from S[0] to S[Nf-1] (S[Nf-1] has a value and not S[Nf])
           moment1 = integrate.simps(integrand[ifminn:ifmax],freq[ifminn:ifmax])     #moment1 = integral1d(freq,integrand,fminn,freq(end));
           # 2: Integral over [freq(end),fmax]
           #    Variance density spectrum in this range: C * f^power, with
           #    C determined by power and spec(end)
           power   = -5  # Power of high-frequency tail
           C       = spec[len(spec)-1] / ( freq[len(freq)-1]**power )    # matlab C       = spec(end) / ( freq(end)^power );
           moment2 = (C / (m+power+1)) * (fmax**(m+power+1) - freq[len(freq)-1]**(m+power+1))   #matlab moment2 = (C / (m+power+1)) * (fmax^(m+power+1) - freq(end)^(m+power+1));
           # Add the two moments
           moment  = moment1 + moment2


    #
    return moment
 
def create_spectrum_jonswap(f=None,fp=None,hm0=None,gammaPeak=3.3,l_fmax=0):
    """
    CREATE_SPECTRUM_JONSWAP  Creates a Jonswap spectrum

    This function creates the Jonswap variance density spectrum, based on a
    given frequency axis, wave height, peak frequency and peak enhancement
    factor.
    Literature: 
    Hasselman, K., e.a. (1973), Erga"nzungsheft zur Deutschen
    Hydrographischen Zeitschrift, Reihe A(8), No. 12
    

    Parameters
    ----------
    f : TYPE
        DESCRIPTION.
    fp : TYPE
        DESCRIPTION.
    hm0 : TYPE
        DESCRIPTION.
    gammaPeak : TYPE
        DESCRIPTION.
    l_fmax : TYPE
        DESCRIPTION.
    f         : array double (1D)
              1D real array containing frequency values. The numbers in
              the array f must be increasing and uniformly spaced 
              (uniform frequency step). Units: Hz
    fp        : double
              peak frequency. Units: Hz
    hm0       : double 
              wave height. Units: m
    gammaPeak : double
              (optional parameter) peak enhancement factor
              Default value is 3.3. No units.
    l_fmax    : double
              optional argument. The imposed spectral wave height Hm0
              holds for the frequency range [f(1),f(end)] (l_fmax = 0, 
              default) or for the frequency range [f(1),inf] (l_fmax =
              1).

    Returns
    -------
    sVarDens = 1D array containing variance density (units m^2/Hz)
    None.

    For l_fmax = 0, the sVarDens is such that integration from f(1) to f(end)
    leads exactly to the given Hm0. 
    For l_fmax = 1, integration from [f(end},inf] is computed using a 
    (-5)-power law. This also means that integration from f(1) to f(end)
    leads to a slightly smaller value for the wave height than the
    prescribed Hm0.

    Syntax:
    sVarDens = createspectrumjonswap(f,fp,hm0,gammaPeak,l_fmax)

    Example 
    -------
    >>> import numpy as np
    >>> f=np.arange(0,2,0.1)  
    >>> Tp = 5.0
    >>> hm0 =1.5
    >>> S = create_spectrum_jonswap(f,1/Tp,hm0,3.3)

    See also create_spectrum_piersonmoskowitz, create_spectrum_tma, 
    spectrum2_timeseries


    """
    
    # --- Ensure array input is of type ndarray.
    f,fSize =engine_core.convert_to_vector(f)
    
    nf = fSize[0]
    
    # Perform check on input arguments
    # --- Check whether input array f is a 1D vector array
    isvalid_size = nf>0     # matlab ( numel(size(f))~=2 || min(size(f)) ~= 1 )
    if not isvalid_size:
       raise ValueError('create_spectrum_jonswap:Input error: Input array f is not 1D ');
    
    if ( l_fmax == 0 ):
       fmax = f[nf-1]
    elif( l_fmax == 1 ):
       fmax = np.inf
    else:
      raise ValueError('create_spectrum_jonswap:Input error:Argument l_fmax must be either 0 or 1')
    
    # Computational core
    # --- Some relevant constants
    sigma_a = 0.07   # Parameter in peak enhancement function
    sigma_b = 0.09   # Parameter in peak enhancement function

    # --- Scaling constant C. 
    # Note that scaling with Hm0 to obtain correct magnitude of S
    # is done further below. The scaling constant C is included for reasons of
    # consistency with formulations as present in literature. For
    # computational reasons, it is not needed.
    g     = 9.81                  # Gravity constant
    alpha = 1                     # Scaling parameter, taken equal to 1. 
    C     = alpha * g**2 * (2*np.pi)**(-4) # Scaling constant

    # --- Initialize variance density spectrum
    sVarDens  = np.zeros( len(f) )

    # --- Evaluate variance density spectrum, for the moment omitting the
    #     weighting
    for iff in np.arange(0,nf):      # matlab 1:length(f);
        f_i = f[iff];
    
        # --- Consider only f_i > 0.
        #     For f_i <=0, the variance density is kept equal to zero
        if ( f_i > np.spacing(1) ):
           # Ratio f/fp
           nu = f[iff] / fp

           # Parameter sigma
           if ( f_i < fp ):
              sigma = sigma_a
           else:
              sigma = sigma_b
        

           # Peak enhancement function
           A = np.exp( -( (nu - 1)**2 ) / (2*sigma**2) )
           lambda_jonswap = gammaPeak**A

           # Variance density
           sVarDens[iff] = C * f_i**(-5) * np.exp( -1.25*nu**(-4) ) * lambda_jonswap

    # --- Compute 'wave height' of the not yet correctly scaled variance 
    #     density spectrum
    m0          =compute_moment( f, sVarDens, 0, f[0], fmax );
    hm0NonScale = 4 * np.sqrt( m0 )

    # --- Perform scaling, to obtain a variance density that has the proper
    #     energy, i.e. corresponding with wave height Hm0
    sVarDens = sVarDens * ( hm0 / hm0NonScale )**2;

    return sVarDens

def create_spectrum_piersonmoskowitz(f=None,fp=None,hm0=None,l_fmax=0):
    """
    
    CREATE_SPECTRUM_PIERSONMOSKOWITZ  Creates a Pierson-Moskowitz spectrum

    This function creates the Pierson-Moskowitz variance density spectrum, 
    based on agiven frequency axis, wave height and peak frequency. The
    Pierson-Moskowitz spectrum is identical to the Jonswap spectrum with a
    peak enhancement factor equal to 1. Furthermore, the Pierson-Moskowitz
    spectrum, the Bretschneider spectrum and the ITTC spectrum are all
    three identical. 
    Literature: 
    Pierson, W.J. and L. Moskowitz (1964). A proposed spectral form for
    fully developed wind seas based on the similarity theory of S.A.
    Kitaigorodskii. Journal of Geophysical Research,Vol. 69, No. 24, pg.
    5181 - 5190. 

    Parameters
    ----------
    f         : array double (1D)
              1D real array containing frequency values. The numbers in
              the array f must be increasing and uniformly spaced 
              (uniform frequency step). Units: Hz
    fp        : double 
              peak frequency. Units: Hz
    hm0       : double
              wave height. Units: m
    l_fmax    :
              optional argument. The imposed spectral wave height Hm0
              holds for the frequency range [f(1),f(end)] (l_fmax = 0, 
              default) or for the frequency range [f(1),inf] (l_fmax =
              1).


    Returns
    -------
    sVarDens : array double (1D)
             1D array containing variance density (units m^2/Hz)

    For l_fmax = 0, the sVarDens is such that integration from f(1) to f(end)
    leads exactly to the given Hm0. 
    For l_fmax = 1, integration from [f(end},inf] is computed using a 
    (-5)-power law. This also means that integration from f(1) to f(end)
    leads to a slightly smaller value for the wave height than the
    prescribed Hm0.


    Syntax:
    sVarDens = create_spectrum_piersonmoskowitz(f,fp,hm0,l_fmax)

    Example:
    >>> import numpy as np
    >>> f=np.arange(0,2,0.01)  
    >>> Tp = 5.0
    >>> hm0 =1.5
    >>> Spm = create_spectrum_piersonmoskowitz(f,1/Tp,hm0)

    See also create_spectrum_jonswap, create_spectrum_tma, 
    spectrum2_timeseries

    """
    
    # --- Ensure array input is of type ndarray.
    f,fSize =engine_core.convert_to_vector(f)
    
    # Computational core
    # --- Use the fact that the Pierson-Moskowitz spectrum is identical to the 
    #     Jonswap spectrum with a peak enhancement factor equal to 1.
    gammaPeak = 1
    sVarDens  = create_spectrum_jonswap(f,fp,hm0,gammaPeak,l_fmax)

    return sVarDens

def compute_tps(f=None,S=None)->float:
    """
    COMPUTE_TPS  Computes smoothed peak period.

    This function computes the smoothed peak period Tps, by means of
    quadratic interpolation, of a given variance density spectrum S = S(f).
    

    Parameters
    ----------
    f    : array double (1D)
         1D array representing frequency axis (unit: Hz)
    S    : array double (1D)
         1D array representing variance density spectrum (units: m2/Hz).


    Returns
    -------
    Tps  : double
         smoothed peak period (units: s)

    Syntax:
    Tps = compute_tps(f,S)


    Example
    >>> import numpy as np
    >>> f=np.arange(0,2,0.01)  
    >>> Tp = 5.0
    >>> hm0 =1.0
    >>> sPM = create_spectrum_piersonmoskowitz(f,1/Tp,hm0)
    >>> Tps = compute_tps(f,sPM)


    See also compute_spectrum_params

    """
    
    # --- Ensure array input is of type ndarray.
    f,fSize =engine_core.convert_to_vector(f)
    S,SSize =engine_core.convert_to_vector(S)
    
    if (fSize[1] >1 or SSize[1]>1):
        raise ValueError('compute_moment: Input error: input should be 1d arrays')
      
    if (fSize[1] >1 and not engine_core.monotonic_increasing_constant_step(f)):
        raise ValueError('compute_moment: Input error: frequency input parameter must be monotonic with constant step size')
    
        
    if (not (fSize[0]==SSize[0])):
        raise ValueError('compute_moment: Input error: array sizes differ in dimension')
        
           
    Smax = max( S )
    if ( Smax < 1e-10 ):
       Tps = -999
       return Tps
    nF   = fSize[0]
    # note: [0] selects first part of the tuple.
    imax = np.where(S==Smax)[0]  # matlab find( S == Smax );
    imax = imax.astype(int)
    nmax = len(imax)

    # --- Depending on value of nF, compute Tps
    if ( nF > 2 ):
       # --- nF > 2 - default situation
       if ( nmax == 1 ):
          # --- nmax = 1
          jmax = imax   # matlab imax   ->in matlab jmax=imax, as imax=np.where() it already account for the fact that indices starts with zero.
          if ( imax == 0 ):
             jmax = 1      # the one after the first one: starting at 0, so jmax must be 1 in python ->matlab jmax=2 
          elif ( imax == nF-1 ):
             jmax = nF-2   # the one before the last one: matlab nF-1
            
          # --- Find polynomial coefficients. note: due to double brackets reduce dimension by selecting [0]
          #ff = np.array([f[jmax-1],f[jmax],f[jmax+1]]).reshape(1,3)[0]
          ff = np.asarray([f[jmax-1],f[jmax],f[jmax+1]]).reshape(1,3)[0]
          ee = np.asarray([S[jmax-1],S[jmax],S[jmax+1]]).reshape(1,3)[0]
          p  = np.polyfit(ff,ee,2);
          a  = p[0]
          b  = p[1]
          # --- Compute Fps
          if ( a < 0. ):
             Fps = -b / (2*a)
          else:
              # Exceptional situation; can only occur if imax=1 or imax=nF
              Fps = f[imax]
          Tps = 1. / Fps
        
       elif ( nmax == 2 ):
            # --- nmax = 2
            if ( (imax[1]-imax[0])==1 ):
               # Points are neighbours
               if ( imax[0] == 0 ):
                  kmax = 1
               elif ( imax[1] == nF-1 ):
                  kmax = nF-2;
               else:
                  kmax = imax[0]
            else:
                # Points are not neighbours - make arbitrary choice
                Tps = 1 / f[imax[0]];
                return Tps
        
            # --- Find polynomial coefficients. note: due to double brackets reduce dimension by selecting [0]
            ff = np.asarray([f[kmax-1], f[kmax], f[kmax+1]]).reshape(1,3)[0]
            ee = np.asarray([S[kmax-1], S[kmax], S[kmax+1]]).reshape(1,3)[0]
            p  = np.polyfit(ff,ee,2)
            a  = p[0]
            b  = p[1]
            # --- Compute Fps (note: in this case, a < 0 always)
            Fps = -b / (2*a)
            Tps = 1 / Fps
       else:
           # --- nmax >= 3 - make arbitrary choice
           Tps=1/f[imax[1]]
    elif ( nF == 2 ):
         # --- nF = 2: two points
         if ( nmax == 1 ):
            # nmax = 1
            Tps = 1 / f[imax]
         else:
            # nmax = 2
            favg = 0.5 * ( f[0] + f[1] )
            Tps = 1 / favg
    
    else:
        # --- nF = 1: one point
        Tps = 1 / f[0]

    return Tps  # TODO: check for some reason single value is stored as array, 

    
def ttsa_compute_spectrum_time_serie(time:np.ndarray=None,quantity:np.ndarray=None, parameter:dict=dict()) ->[ np.ndarray,np.ndarray, float,float,float,float,float]:
    
    if (time is None) or not(isinstance(time,np.ndarray)):
       raise ValueError('Input parameter time must be of type ndarray')
       
    if (quantity is None) or not(isinstance(quantity,np.ndarray)):
       raise ValueError('Input parameter quantity must be of type ndarray')
    
    if (parameter ==None) or not (isinstance(parameter,dict)):
       raise ValueError('Input parameter quantity must be of type ndarray') 
    
       
    df   = parameter.setdefault('DF',0.01)
    data_type_window = parameter.setdefault('DATA_TYPE_WINDOW',DataWindowTypes.HANNING) 
    number_freq_bins = int(parameter.setdefault('NUMBER_FREQ_BINS',1))
    
    # --- ensure that total number of frequency bins is even 
    if not(number_freq_bins % 2 ==0):
         number_freq_bins = max(number_freq_bins-1,1)
     
    number_time_windows = int(parameter.setdefault('NUMBER_TIME_WINDOWS',1))
    freq_range = parameter.setdefault('FREQ_RANGE',None) 
       
    
    # --- Evaluate some relevant parameters
    #     dt           : time step
    #     number_samples_per_win: number of time samples per window
    dt             = time[1] - time[0];    
    number_samples_per_win = int(len(quantity) / number_time_windows)
    
    # --- ensure number of time windows per interval are even.
    if not(number_samples_per_win % 2 ==0):
         number_samples_per_win = max(number_samples_per_win-1,1)
    
    # --- Call to routine that computes spectrum and additional output
    #     parameters
    [freqs, spectrum,hm0,tp,tm_10,tm01,tm02] = ttsa_specgk(quantity=quantity,number_samples_per_win= number_samples_per_win,number_freq_bins=number_freq_bins, \
                                                           dt=dt,data_type_window=data_type_window,freq_range=freq_range)

    return [freqs,spectrum,hm0,tp,tm_10,tm01,tm02]


def ttsa_specgk(quantity:np.ndarray=None,number_samples_per_win:int=None, \
                number_freq_bins:int=1,dt:float=None,data_type_window:DataWindowTypes=DataWindowTypes.HANNING,\
                  freq_range:np.ndarray=None) ->[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray, np.ndarray]:
    
    # specgk   Power spectrum computation, with smoothing in the frequency domain
    #
    # Input:
    #    Y  contains the data. Length(Y) should be even. 
    #    N  is the number of samples per data segment.
    #    M  is the number of frequency bins over which is smoothed.
    #       No smoothing for M=1 (default)
    #    DT is the time step
    #   DW is the data window type (optional): DW = 1 for Hann window (default)
    #                                           DW = 2 for rectangular window
    #    rang gives frequency ranges over which wave parameters are computed
    #
    # Output:
    #    P     contains the spectral estimates
    #    F     contains the frequencies at which P is given
    #    Hm0   contains wave height
    #    Tp    contains peak period
    #    Tm_10 contains T_{m-1,0}
    #    Tm01  contains T_{m01}
    #    Tm02  contains T_{m02}
    #
    # Gert Klopman, Delft Hydraulics, 1994
    # (changed (slightly) by M.P.C. de Jong 1999)
    # (changed (slightly) by I. Wenneker 2006)

    
    if (quantity is None) or not(isinstance(quantity,np.ndarray)):
       raise ValueError('Input parameter quantity must be of type ndarray')
       
    if (number_samples_per_win ==None) or not(isinstance(number_samples_per_win,int)) or not(number_samples_per_win>0):
       raise ValueError('Input parameter number_samples_per_win must be of type int and larger than zero')
       
    if (number_freq_bins == None) or not(isinstance(number_freq_bins,int)) or not(number_freq_bins>0):
       raise ValueError('Input parameter nbin must be of type int and larger than zero')
    
    if not(dt==None) and not(isinstance(dt,float)) and not(dt>0.0):
       raise ValueError('Input parameter dt must be of type float and larger than zero') 
   
    if (data_type_window ==None) or not (isinstance(data_type_window,DataWindowTypes)) :
       raise ValueError('Input parameter dwin must be of type DataWindowTypes') 
    
    if not (freq_range==None) and not (isinstance(freq_range,np.ndarray)):
        raise ValueError('Input parameter dwin must be of type ndarray') 
    
    if not freq_range==None:
       [dimx_freq_range,dimy_freq_range] =engine_core._size(freq_range)
       if not (dimy_freq_range==2):
          raise ValueError('Second dimension of the frequency range must be two:{0}'.format(dimy_freq_range)) 
    
    y,quantity_size =engine_core.convert_to_vector(quantity)
        
    # ensure that length of Y is even
    #Ny = 2 * floor( length(Y) / 2 );
    #Y  = Y(1:Ny);
    #if ( N == Ny + 1 )
    #    N = Ny;
    #end
    
    # --- see specgk in ttsa , if nargin<4 -> dt=1 
    if dt ==None:
       dt =1.0  
        
   
    df = 1.0 / (number_samples_per_win * dt) ;

    # data window

    w = np.zeros(len(y),dtype=float)
    if data_type_window ==DataWindowTypes.HANNING:
      # Hann
      w  = np.hanning(number_samples_per_win) # hydMeas_hanning(N);
      dj = int(number_samples_per_win/2);
    else:
      # rectangle
      w  =np.ones(number_samples_per_win,dtype = float);
      dj = number_samples_per_win

    varw = sum (w**2) / number_samples_per_win 

    # summation over segments

    ny    = quantity_size[0]   #max(len(y))
    avg   = sum(y) / ny
    p     = np.zeros(len(w),dtype=float)
    pavg  = np.zeros(len(w),dtype=float)
    ns    = 0
    
    # for j=[1:dj:ny-N+1],
    for j in np.arange(0,ny-number_samples_per_win+1,dj):
        ns = ns + 1;

        #   compute FFT of signal
        idx = np.arange(j,j+number_samples_per_win) #p = Y([j:j+N-1]') - avg;
        #p = Y([j:j+N-1]) - avg;
        p  = y[idx] -avg
        p = w * p 
        # --- NOTE setting relative tolerance :It compares the difference between actual and desired to atol + rtol * abs(desired).
        #     the fft from numpy and scipy differ from the matlab fft (
        #  -https://stackoverflow.com/questions/44677001/fft-results-matlab-vs-numpy-python-not-the-same-results
        #  -http://www.bmtfx.com/matlab-vs-python-spectrum/?lang=en, 
        #  -https://pythonnumericalmethods.berkeley.edu/notebooks/chapter24.04-FFT-in-Python.html)
        #p= fft(p) 
        p = np.fft.fft(p) 
        # compute periodogram

        pavg = pavg + p * np.conj(p) 


    pavg = (2.0 / (ns * (number_samples_per_win**2) * varw * df)) * pavg 

    # --- smoothing
    if number_freq_bins>1:
       w = np.hanning(number_freq_bins)  # w = hydMeas_hanning(M);
       w = w / sum(w)
       #w = [w(ceil((M+1)/2):M); zeros(N-M,1); w(1:ceil((M+1)/2)-1)];
       
       # --- assuming number of freq_bins is even 
       if number_freq_bins%2==0:
          left_side_number_smoothing_bins =round(number_freq_bins/2)
          right_side_number_smoothing_bins=left_side_number_smoothing_bins
       else:
          left_side_number_smoothing_bins =round(number_freq_bins/2)
          right_side_number_smoothing_bins=left_side_number_smoothing_bins-1 
           
       part_a = w[ left_side_number_smoothing_bins:number_freq_bins]
       part_b = np.zeros((number_samples_per_win-number_freq_bins),dtype=float)
       part_c = w[0:right_side_number_smoothing_bins]
       w = np.concatenate((part_a,part_b, part_c))
       # w = np.arange(w[np.ceil((number_freq_bins+1)/2):number_freq_bins], \
       #                np.zeros((number_samples_per_win-number_freq_bins,1),dtype=float), \
       #                w[1:np.ceil((number_freq_bins+1)/2)-1])
       w = np.fft.fft(w)
       pavg = np.fft.fft(pavg);
       pavg = np.fft.ifft(w * pavg);
    
    
    P = abs(pavg[0:int(number_samples_per_win/2)]);

    # frequencies
    #F = ([1:1:N/2]' - 1) * df;
    F = np.arange(0,int(number_samples_per_win/2)) * df
    if freq_range==None:
       freq_range = np.asarray([0,F[-1]]).reshape(1,2)  # F[-1] = F[number_freq_bins/2] 
       dimx_freq_range =1 
       
       
    if data_type_window == DataWindowTypes.HANNING:
       nn = (ns + 1) * int(number_samples_per_win/ 2)
    else:
       nn = ns * number_samples_per_win

    # --- Compute wave parameters in given spectral ranges
     

    hm0   = np.zeros(dimx_freq_range,dtype=float)
    tm_10 = np.zeros(dimx_freq_range,dtype=float)
    tm01  = np.zeros(dimx_freq_range,dtype=float)
    tm02  = np.zeros(dimx_freq_range,dtype=float)
    tp    = np.zeros(dimx_freq_range,dtype=float)
   
     
    # calculation of average and signal variance
    
    # avg = sum (Y(1:nn)) / nn;
    avg = sum (y[0:nn]) / nn
    # var = sum ((Y(1:nn) - avg).^2) / (nn - 1);
    var = sum ((y[0:nn] - avg)**2) / (nn - 1)
 
    # If wave signal is zero, then the operations below are not useful anymore
    # (in fact, lead to divisions by zero). 
    # Therefore, we set the output values to zero
    if ( var < 1e-10 ):
       return;


    # --- Compute wave parameters in given spectral ranges
    #m0     = (0.5 * P(1) + sum(P(2:N/2-1)) + 0.5 * P(N/2)) * df;
    m0     = (0.5 * P[0] + sum(P[1:int(number_samples_per_win/2)-2]) + 0.5 * P[int(number_samples_per_win/2)-1] )* df;
    err_m0 = m0 / var -1

    # correct spectral densities to get right m0 (= variance)
    P = P * (var / m0)


    for irang in np.arange(0,dimx_freq_range):
       # --- Take minimal and maximal frequencies
       
       f_min = freq_range[irang][0]
       f_min = max( f_min, 0.0      );
       f_max = freq_range[irang][1]; 
       f_max = min( f_max, F[int(number_samples_per_win/2)-1])
    
       # --- Find integers corresponding to which cell f_min and f_max belong
       i_min = round( f_min / df );
       i_max =  round( f_max / df );
    
       # m0: Integrate from f_min to f_max
       m0 = ( F[i_min] + 0.5*df - f_min ) * P[i_min];
       m0 = ( f_max - F[i_max] + 0.5*df ) * P[i_max] + m0;
       m0 = m0 + df * sum( P[i_min+1:i_max-1] ) 

       # m_1: Integrate from f_min to f_max
       if ( i_min == 0 ):
          m_1 = 0.  # Omit division by zero (F(1) = 0)
       else:
          m_1 = ( F[i_min] + 0.5*df - f_min ) * ( P[i_min] / F[i_min] )
    
       m_1 = ( f_max - F[i_max] + 0.5*df ) * ( P[i_max] / F[i_max] ) + m_1
       m_1 = m_1 + df * sum( P[i_min+1:i_max-1] / F[i_min+1:i_max-1] )

       # m1: Integrate from f_min to f_max
       m1 = ( F[i_min] + 0.5*df - f_min ) * ( P[i_min] * F[i_min] );
       m1 = ( f_max - F[i_max] + 0.5*df ) * ( P[i_max] * F[i_max] ) + m1;
       m1 = m1 + df * sum( P[i_min+1:i_max-1] * F[i_min+1:i_max-1] ); 

       # m1: Integrate from f_min to f_max
       m2 = ( F[i_min] + 0.5*df - f_min ) * ( P[i_min] * F[i_min]**2 );
       m2 = ( f_max - F[i_max] + 0.5*df ) * ( P[i_max] * F[i_max]**2 ) + m2;
       m2 = m2 + df * sum( P[i_min+1:i_max-1] * F[i_min+1:i_max-1]**2 ); 

       # --- Evaluate wave parameters
       hm0[irang]   = 4.0 * np.sqrt( m0 )
       tm01[irang]  = m0 / m1
       tm_10[irang] = m_1 / m0
       tm02[irang]  = np.sqrt( m0 / m2 )
    
       #[dummy,IP]   = max( P[i_min+1:i_max-1] )
       max_value = max( P[i_min+1:i_max-1] )
       IP = P[i_min+1:i_max-1].argmax()
       tp[irang]    = 1.0 / F[IP + i_min+1];  # not IP is calculated frim i_min+1...       
       return F,P,hm0,tp,tm_10,tm01,tm02

def compute_spectrum_time_serie(t=None,xTime=None,dfDesired=None) ->[ np.ndarray,np.ndarray ]:
    """
    @brief
    COMPUTE_SPECTRUM_TIME_SERIE  Computes variance density spectrum from given time
    series  (WAVELAB : computespectrum1)

    @section description_xxx Description 
    This function computes a variance density spectrum sCoarse = sCoarse(fCoarse)
    on a frequency axis fCoarse from a given surface elevation time series
    xTime = xTime(t), with time axis t. The frequency spacing of the output
    spectrum is given by dfDesired.
    

    Parameters
    ----------
    @param t  : array double (1D)
              time array [s]
    @param xTime     : array double (1D)
              time series of surface elevation
    @param dfDesired : double
              (optional parameter) desired frequency spacing in Hertz on 
              which sCoarse must be computed. If this input parameter is
              omitted, then dfDesired is determined automatically, and is
              based on the length of the time series


    Returns
    -------
    fCoarse  : array double (1D) 
             frequency axis of computed spectrum. The frequency
             spacing is (close to) dfDesired
    sCoarse  : array double (1D)
             variance density spectrum


    Syntax:
    [fCoarse,sCoarse] = compute_spectrum_time_serie(t,xTime,dfDesired);

    Example
    >>> import numpy as np
    >>> dt =0.1
    >>> t = np.arange(0,1000+dt,dt)  # Time axis
    >>> z = np.sin(t) + np.cos(2*t)  # Surface elevation data
    >>> df = 0.01                    # Choose value for frequency axis
    >>> [freq,varDens] = compute_spectrum_time_serie(t,z,df)
 
    See also 

    """
    
    # --- Transform to frequency domain ( input check is done in time2freq_nyquist)
    [f,xFreq,isOdd]   = time2freq_nyquist(t,xTime)
    df                = (f[1] - f[0])
    Ntime             = len( t )
    sFine             = 2*xFreq*np.conj(xFreq)/(df * Ntime * Ntime )
    sFine             = sFine.real

    # --- Perform averaging
    if ( dfDesired is None ):
       [fCoarse,sCoarse] = frequency_averaging(f,sFine)
    else:
       [fCoarse,sCoarse] = frequency_averaging(f,sFine,dfDesired)    

    return [fCoarse, sCoarse]

def compute_spectrum_freq_serie(f=None,xFreq=None,dfDesired=None,Ntime=None):
    """
    COMPUTE_SPECTRUM_FREQ_SERIE Computes variance density spectrum from given complex
    spectrum of Fourier components  (WAVELAB: computespectrum2)

    This function computes a variance density spectrum sCoarse = sCoarse(fCoarse)
    on a frequency axis fCoarse from a given complex spectrum xFreq = xFreq(f)
    of Fourier coefficients on a frequency axis f. The frequency spacing of 
    the output spectrum is given by dfDesired.

    Parameters
    ----------
    f         : array double (1D) 
              frequency axis of input spectrum [Hz]
    xFreq     : array double (1D)
              wave spectrum of complex Fourier coefficients
    dfDesired : double
              desired frequency spacing in Hertz on which sCoarse must be
              computed
    Ntime     : integer 
              number of time elements in original time signal


    Returns
    -------
    fCoarse   :array double (1D)
              frequency axis of computed spectrum. The frequency
              spacing is (close to) dfDesired
    sCoarse   : array double (1D)
              variance density spectrum


    Syntax:
          [fCoarse,sCoarse] = compute_spectrum_freq_serie(f,xFreq,dfDesired,Ntime)

    Example:
    >>> import numpy as np
    >>> dt =0.1
    >>> t = np.arange(0,1000+dt,dt)  # Time axis
    >>> z = np.sin(t) + np.cos(2*t)  # Surface elevation data
    >>> [f,xFreq,isOdd] = time2freq_nyquist(t,z)
    >>> df = 0.01                    # Choose value for frequency axis
    >>> [fS,S] = compute_spectrum_time_serie(t,xFreq,df)
 
    See also 

    """
    f,fSize =engine_core.convert_to_vector(f)
    xFreq,xFreqSize =engine_core.convert_to_vector(xFreq)
    
    if (fSize[1] >1 or xFreqSize[1]>1):
        raise ValueError('compute_spectrum_freq_serie: Input error: input should be 1d arrays')
      
    if (not engine_core.monotonic_increasing_constant_step(f)):
        raise ValueError('compute_spectrum_freq_serie: Input error: frequency input parameter must be monotonic with constant step size')
    
        
    if (not (fSize[0]==xFreqSize[0])):
        raise ValueError('compute_spectrum_freq_serie: Input error: array sizes differ in dimension')
    
    if (Ntime==None):
        raise ValueError('compute_spectrum_freq_serie: Input error: Number of time samples not specified')
    
    # --- Transform to frequency domain
    df    = (f[1] - f[0])
    sFine = 2*xFreq*np.conj(xFreq)/(df * Ntime * Ntime )

    # --- Perform averaging
    [fCoarse,sCoarse] = frequency_averaging(f,sFine,dfDesired)

    return [fCoarse,sCoarse]

def test_doctstrings():
    import doctest
    doctest.testmod()
    
if __name__ == '__main__':
    test_doctstrings()
     
    #isOdd = 1;
    #f = [0,0.4, 0.8, 1.2, 1.62,2.0]
    #xFreq = [6.3962 + 0j,0.7636 + 0.3877j,-1.1051 - 0.0730j,1.1387 - 0.9262j, 0.5616 - 1.5596j,-0.0759 - 0.3736j];

    # # --- Apply function
    #[fTot,xFreqTot] = unfold_spectrum(f,xFreq,isOdd)
    

