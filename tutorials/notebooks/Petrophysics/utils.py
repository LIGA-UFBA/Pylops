import numpy as np
from typing import Union
import matplotlib.pyplot as plt

def create_mask_value(m: np.ndarray,
                      value: float) -> np.ndarray:
    """
    Create a binary mask from a model array by thresholding.

    Parameters
    ----------
    m : :obj:`numpy.ndarray`
        Input array (e.g., velocity model).
    value : float
        Threshold value. Elements strictly greater than `value` are set to 1, otherwise 0.

    Returns
    -------
    :obj:`numpy.ndarray`
        Integer mask with the same shape as `m`, containing 0s and 1s.
    """
    mask = m > value
    mask = mask.astype(int)
    return mask


def Wiener_Filt(wav_orig: np.ndarray,
                wav_targ: np.ndarray, 
                orig_data: np.ndarray, 
                eps: float = 1e-4) -> np.ndarray:
    """
    Apply a frequency-domain Wiener shaping filter to map data from an original
    source wavelet to a target wavelet.

    The shaping filter is constructed as:
        F_w(f) = W_t(f) * conj(W_o(f)) / ( |W_o(f)|^2 + eps^2 )
    where W_o(f) and W_t(f) are the FFTs of the original and target wavelets, respectively.

    Parameters
    ----------
    wav_orig : :obj:`numpy.ndarray`
        Original source wavelet (1D), length `nt`.
    wav_targ : :obj:`numpy.ndarray`
        Target source wavelet (1D), length `nt`.
    orig_data : :obj:`numpy.ndarray`
        Input data organized as a list/array of shots, each of shape (ntraces, nt) or (nt, ntraces).
        This function internally reorders to (nshots, nt, ntraces).
    eps : float, optional
        Stabilization term (pre-whitening) for the Wiener denominator. Default is 1e-4.

    Returns
    -------
    :obj:`numpy.ndarray`
        Filtered data with the same external organization as `orig_data`:
        shape (nshots, ntraces, nt), but with wavelet shaped to `wav_targ`.

    Notes
    -----
    - Uses zero-padding to `nt_o = 2*nt` for better FFT handling.
    - FFTs are computed with `np.fft.rfft(..., norm='ortho')`.
    - Filtering is done shot-wise and per-trace in the frequency domain.
    """
    dobs_reorg = np.asarray([shot.T for shot in orig_data])  # (ns, nt, nx)
    ns, nt, nx = dobs_reorg.shape
    nt_o = 2 * nt

    # Zero-pad data and wavelets
    x_ext = np.zeros((ns, nt_o, nx))
    x_ext[:, :nt, :] = dobs_reorg
    wo_ext = np.zeros(nt_o); wo_ext[:nt] = wav_orig
    wt_ext = np.zeros(nt_o); wt_ext[:nt] = wav_targ

    # Wavelet spectra (one-sided FFT)
    FT_wo = np.fft.rfft(wo_ext, n=nt_o, norm='ortho')
    FT_wt = np.fft.rfft(wt_ext, n=nt_o, norm='ortho')

    # Wiener shaping filter Fw
    denom = (np.abs(FT_wo) ** 2) + (eps ** 2)
    Fw = (FT_wt * np.conj(FT_wo)) / denom  # (nf,)

    # Apply shaping filter to each shot/trace
    filtered_list = []
    for i in range(ns):
        FT_x = np.fft.rfft(x_ext[i], n=nt_o, axis=0, norm='ortho')   # (nf, nx)
        FT_y = Fw[:, None] * FT_x                                    # (nf, nx)
        y_ext = np.fft.irfft(FT_y, n=nt_o, axis=0, norm='ortho').real
        filtered_list.append(y_ext[:nt, :])

    filtered = np.asarray(filtered_list)                              # (ns, nt, nx)
    filtered_reorg = np.asarray([shot.T for shot in filtered])        # (ns, nx, nt)
    return filtered_reorg


def plot_data(data: np.ndarray, 
              shot: int = 0,
              cmap: str = 'gray',
              title: str = 'Data') -> None:
    """
    Plot a single shot gather with automatic robust scaling.

    Parameters
    ----------
    data : :obj:`numpy.ndarray`
        Seismic data with shape (nshots, ntraces, nt) or compatible.
    shot : int, optional
        Index of the shot to plot. Default is 0.
    cmap : str, optional
        Matplotlib colormap. Default is 'gray'.
    title : str, optional
        Figure title.

    Returns
    -------
    None

    Notes
    -----
    The color scale is set using the 2nd and 98th percentiles computed across the entire dataset
    to obtain robust vmin/vmax.
    """
    d_vmin, d_vmax = np.percentile(np.hstack(data).ravel(), [2, 98])
    plt.imshow(data[shot].T, aspect='auto', cmap=cmap, vmin=d_vmin, vmax=d_vmax)
    plt.ylabel('Time (ms)', fontsize=14)
    plt.xlabel('Trace Number', fontsize=14)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.colorbar()
    plt.show()

def get_alfa_g(yk: np.ndarray, 
               sk: np.ndarray) -> float:
    """
    Compute a step length using Barzilai–Borwein (BB) heuristics and select
    between BB1 and BB2 based on a safeguard.

    Parameters
    ----------
    yk : :obj:`numpy.ndarray`
        Difference of gradients, typically y_k = g_{k+1} - g_k (flattened internally).
    sk : :obj:`numpy.ndarray`
        Difference of iterates, typically s_k = x_{k+1} - x_k (flattened internally).

    Returns
    -------
    float
        Step length `alfa` chosen as:
        - BB2 = (sᵀy)/(yᵀy) if 0 < (BB2/BB1) < 1
        - else BB1 = (sᵀs)/(sᵀy)

    Notes
    -----
    - BB1 (long step) and BB2 (short step) are classic choices in gradient methods.
    - This selection rule is a common safeguard to avoid overly aggressive steps.
    """
    sk = sk.reshape(-1)
    yk = yk.reshape(-1)

    term1 = np.dot(sk, sk)   # s^T s
    term2 = np.dot(sk, yk)   # s^T y
    term3 = np.dot(yk, yk)   # y^T y

    abb1 = term1 / term2     # BB1
    abb2 = term2 / term3     # BB2
    abb3 = abb2 / abb1

    if abb3 > 0 and abb3 < 1:
        alfa = abb2
    else:
        alfa = abb1
    return alfa

def Frequency_spectrum(wavelet: np.ndarray, 
                       dt: float, 
                       tol: float = 1e-1) -> tuple:
    """
    Compute the one-sided amplitude spectrum of a wavelet and return its peak 
    frequency and maximum non-zero frequency.

    Parameters
    ----------
    wavelet : :obj:`numpy.ndarray`
        Time-domain wavelet (1D array).
    dt : float
        Sampling interval in seconds.
    tol : float, optional
        Tolerance for considering amplitude as non-zero (default=1e-1).

    Returns
    -------
    freqs : :obj:`numpy.ndarray`
        Frequency axis (Hz) for the one-sided spectrum.
    amp : :obj:`numpy.ndarray`
        Amplitude spectrum |W(f)|.
    f_peak : float
        Peak frequency in Hz (frequency of maximum amplitude).
    f_max : float
        Maximum frequency with non-zero amplitude (within tolerance).
    """
    w = np.asarray(wavelet).ravel().astype(float)
    n = w.size

    W = np.fft.rfft(w)                    # complex spectrum
    f = np.fft.rfftfreq(n, d=dt)          # frequency axis (Hz)

    amp = np.abs(W)                       # amplitude spectrum

    idx_peak = np.argmax(amp)
    f_peak = f[idx_peak]

    idx_fmax = np.where(amp > tol)[0]  # taking max frequency
    f_max = f[idx_fmax[-1]] if idx_fmax.size > 0 else 0.0

    print(f"Peak Frequency: {round(f_peak)} Hz | Max Frequency: {round(f_max)} Hz")
    
    plot_freq_spectrum(wavelet, f, amp, f_peak, f_max, int(idx_fmax[-1]))
    
    return f, amp, f_peak, f_max


def plot_freq_spectrum(wavelet: np.ndarray, 
                       freqs: Union[tuple, np.ndarray], 
                       amp: np.ndarray,
                       fpeak: Union[int, float], 
                       fmax: float, 
                       idx_fmax: int):
    """
    Plot the time-domain wavelet and its one-sided amplitude spectrum, highlighting key frequency markers.

    Parameters
    ----------
    wavelet : np.ndarray
        Time-domain wavelet (1D array).
    freqs : tuple or np.ndarray
        Frequency axis (Hz), must align with the `amp` array.
    amp : np.ndarray
        One-sided amplitude spectrum |W(f)| of the wavelet.
    fpeak : int or float
        Peak frequency (Hz) to highlight with a vertical line and label.
    fmax : float
        Maximum frequency limit (Hz) for plotting the spectrum.
    idx_fmax : int
        Index corresponding to the maximum frequency (`fmax`) in the `freqs` array.

    Returns
    -------
    None

    Notes
    -----
    - The wavelet is plotted in the time domain using its non-zero segment.
    - The frequency spectrum is plotted up to `2 * idx_fmax` for clarity.
    - The peak and maximum frequencies are highlighted with markers and labels.
    - Vertical line is drawn at `round(fpeak)` for visibility.
    """
    
    begin = np.where(wavelet > 0)[0][0]
    end = np.where(wavelet > 0)[-1][-1]
    idx_peak = np.argmax(amp) # peak frequency index

    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.plot(wavelet[begin-begin:end+begin], color='black')
    plt.title(f'Wavelet with {round(fpeak)} Hz')

    # Frequency-domain
    plt.subplot(1, 2, 2)
    plt.plot(freqs[:idx_fmax*2], amp[:idx_fmax*2], color='black')
    plt.scatter(freqs[idx_peak], amp[idx_peak], color="red", marker="o", s=60, label=f"Peak Frequency: {round(fpeak)} Hz")
    plt.scatter(freqs[idx_fmax], amp[idx_fmax], color="orange", marker="o", s=60, label=f"Max Frequency: {round(fmax)} Hz")
    plt.vlines(x=round(fpeak), ymin=amp.min(), ymax=amp.max(), color='red')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title("Frequency Spectrum")
    plt.legend()
    plt.tight_layout()
    plt.show()
