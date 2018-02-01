from devito import Dimension
from devito.function import SparseTimeFunction
from devito.logger import error

import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    plt = None

__all__ = ['PointSource', 'Receiver', 'Shot', 'RickerSource', 'GaborSource']


class PointSource(SparseTimeFunction):
    """Symbolic data object for a set of sparse point sources

    :param name: Name of the symbol representing this source
    :param grid: :class:`Grid` object defining the computational domain.
    :param coordinates: Point coordinates for this source
    :param data: (Optional) Data values to initialise point data
    :param ntime: (Optional) Number of timesteps for which to allocate data
    :param npoint: (Optional) Number of sparse points represented by this source
    :param dimension: :(Optional) class:`Dimension` object for
                       representing the number of points in this source

    Note, either the dimensions `ntime` and `npoint` or the fully
    initialised `data` array need to be provided.
    """
    def __init__(self, *args, **kwargs):
        if not self._cached():
            p_dim = kwargs.get('dimension', Dimension(name='p_%s' % kwargs.get('name')))
            npoint = kwargs.get('npoint')
            ntime = kwargs.get('ntime', None)
            if kwargs.get('data', None) is None:
                if ntime is None:
                    error('Either data or ntime are required to'
                          'initialise source/receiver objects')
            else:
                ntime = ntime or data.shape[0]
            kwargs['nt'] = ntime
            kwargs['dimensions'] = (kwargs.get('grid').time_dim, p_dim)
            kwargs['shape'] = (ntime, npoint)
            super(PointSource, self).__init__(*args, **kwargs)
            
            if kwargs.get('data', None) is not None:
                self.data[:] = data


Receiver = PointSource
Shot = PointSource


class WaveletSource(PointSource):
    """
    Abstract base class for symbolic objects that encapsulate a set of
    sources with a pre-defined source signal wavelet.

    :param name: Name for the resulting symbol
    :param grid: :class:`Grid` object defining the computational domain.
    :param f0: Peak frequency for Ricker wavelet in kHz
    :param time: Discretized values of time in ms
    """

    def __init__(self, *args, **kwargs):
        if not self._cached():
            self.time = kwargs.get('time')
            self.npoint = kwargs.get('npoint', 1)
            self.f0=f0 = kwargs.get('f0')
            kwargs['ntime'] = len(self.time)
            kwargs['npoint'] = self.npoint
            super(WaveletSource, self).__init__(*args, **kwargs)
            for p in range(kwargs['npoint']):
                self.data[:, p] = self.wavelet(self.f0, self.time)

    def wavelet(self, f0, t):
        """
        Defines a wavelet with a peak frequency f0 at time t.

        :param f0: Peak frequency in kHz
        :param t: Discretized values of time in ms
        """
        raise NotImplementedError('Wavelet not defined')

    def show(self, idx=0, time=None, wavelet=None):
        """
        Plot the wavelet of the specified source.

        :param idx: Index of the source point for which to plot wavelet
        :param wavelet: Prescribed wavelet instead of one from this symbol
        :param time: Prescribed time instead of time from this symbol
        """
        wavelet = wavelet or self.data[:, idx]
        time = time or self.time
        plt.figure()
        plt.plot(time, wavelet)
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude')
        plt.tick_params()
        plt.show()


class RickerSource(WaveletSource):
    """
    Symbolic object that encapsulate a set of sources with a
    pre-defined Ricker wavelet:

    http://subsurfwiki.org/wiki/Ricker_wavelet

    :param name: Name for the resulting symbol
    :param grid: :class:`Grid` object defining the computational domain.
    :param f0: Peak frequency for Ricker wavelet in kHz
    :param time: Discretized values of time in ms
    """

    def wavelet(self, f0, t):
        """
        Defines a Ricker wavelet with a peak frequency f0 at time t.

        :param f0: Peak frequency in kHz
        :param t: Discretized values of time in ms
        """
        r = (np.pi * f0 * (t - 1./f0))
        return (1-2.*r**2)*np.exp(-r**2)


class GaborSource(WaveletSource):
    """
    Symbolic object that encapsulate a set of sources with a
    pre-defined Gabor wavelet:

    https://en.wikipedia.org/wiki/Gabor_wavelet

    :param name: Name for the resulting symbol
    :param grid: :class:`Grid` object defining the computational domain.
    :param f0: Peak frequency for Ricker wavelet in kHz
    :param time: Discretized values of time in ms
    """

    def wavelet(self, f0, t):
        """
        Defines a Gabor wavelet with a peak frequency f0 at time t.

        :param f0: Peak frequency in kHz
        :param t: Discretized values of time in ms
        """
        agauss = 0.5 * f0
        tcut = 1.5 / agauss
        s = (t-tcut) * agauss
        return np.exp(-2*s**2) * np.cos(2 * np.pi * s)
