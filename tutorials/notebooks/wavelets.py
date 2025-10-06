import numpy as np
from devito import Grid
from examples.seismic import TimeAxis, RickerSource

def Ricker(nt: int, 
           dt: float, 
           fpeak: float, 
           dtype=np.float32) -> np.ndarray:
    """
    Build a wavelet (cosine-modulated Gaussian) consistent with the Fortran implementation.

    Parameters
    ----------
    nt : int
        Number of time samples.
    dt : float
        Sampling interval in seconds.
    fpeak : float
        Peak frequency in Hz.
    dtype : :obj:`numpy.dtype`, optional
        Output data type (default is float32 to match Fortran `real`).

    Returns
    -------
    :obj:`numpy.ndarray`
        Generated wavelet samples as a 1D array with length `nt`.
    """
    if fpeak <= 0:
        raise ValueError("fpeak must be > 0")
    t = np.arange(nt, dtype=np.float64) * dt
    wpeak = 2.0 * np.pi * fpeak
    waux  = 0.5 * wpeak
    tdelay = 6.0 / (5.0 * fpeak)
    tt = t - tdelay
    fonte = np.exp(-(waux * waux) * (tt * tt) / 4.0) * np.cos(wpeak * tt)
    return fonte.astype(dtype, copy=False)


def Ricker_Devito(shape: tuple, 
                  spacing: tuple, 
                  origin: tuple, 
                  dt:float, 
                  nt: int,
                  fpeak: float,
                  t0: float) -> np.ndarray:
    """
    Build a Devito Ricker source and return the time series.

    Parameters
    ----------
    shape : :obj:`tuple` or :obj:`numpy.ndarray`
        Model shape ``(nx, nz)``.
    spacing : :obj:`tuple` or :obj:`numpy.ndarray`
        Grid spacing ``(dx, dz)`` in meters.
    origin : :obj:`tuple` or :obj:`numpy.ndarray`
        Grid origin ``(ox, oz)`` in meters.
    dt : float
        Time sampling in seconds (will be converted to milliseconds for Devito's TimeAxis).
    nt : int
        Number of time samples.
    fpeak : float
        Peak frequency in Hz (will be converted to kHz for Devito's RickerSource).
    t0 : float or None
        Wavelet delay in milliseconds for Devito. If None, Devito uses 1/f0 (in ms).

    Returns
    -------
    :obj:`numpy.ndarray`
        Wavelet samples as a 1D array with length `nt`.

    Notes
    -----
    - Devito's `TimeAxis` expects milliseconds; we pass `dt*1e3`.
    - Devito's `RickerSource` expects frequency in kHz; we pass `fpeak/1e3`.
    - The returned array is `src.data[:]` reshaped to 1D (length `nt`).
    """
    extent = (shape[0]*spacing[0], shape[1]*spacing[1])
    grid = Grid(shape=shape, extent=extent, origin=origin)
    time_range = TimeAxis(start=0.0, step=dt*1e3, num=nt)

    src = RickerSource(name="src", grid=grid, f0=fpeak/1e3,
                       time_range=time_range, a=1.0, t0=t0)
    return src.data[:].reshape(nt)