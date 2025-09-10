import numpy as np
import matplotlib.pyplot as plt
from pylops.waveeqprocessing.twoway import AcousticWave2D


def create_mask_value(m, value):
    mask = m > value
    mask = mask.astype(int)
    return mask

def Wiener_Filt(wav_orig, wav_targ, orig_data, eps = 1e-4):

    dobs_reorg = np.asarray([shot.T for shot in orig_data])
    ns,nt, nx = dobs_reorg.shape
    nt_o = 2 * nt 

    x_ext = np.zeros((ns, nt_o, nx))
    x_ext[:, :nt, :] = dobs_reorg
    wo_ext = np.zeros(nt_o); wo_ext[:nt] = wav_orig
    wt_ext = np.zeros(nt_o); wt_ext[:nt] = wav_targ

    FT_wo = np.fft.rfft(wo_ext, n=nt_o, norm='ortho')
    FT_wt = np.fft.rfft(wt_ext, n=nt_o, norm='ortho')

    denom = (np.abs(FT_wo) ** 2) + (eps ** 2)
    Fw = (FT_wt * np.conj(FT_wo)) / denom

    filter = []
    for i in range(ns):
        FT_x = np.fft.rfft(x_ext[i], n=nt_o, axis=0, norm='ortho')  # (nf, nx)
        FT_y = Fw[:, None] * FT_x
        y_ext = np.fft.irfft(FT_y, n=nt_o, axis=0, norm='ortho').real
        filter.append(y_ext[:nt, :])

    filtered = np.asarray(filter)
    filtered_reorg = np.asarray([shot.T for shot in filtered])
    return filtered_reorg

def plot_data(data, shot=0, cmap='gray', title='Data'):
    d_vmin, d_vmax = np.percentile(np.hstack(data).ravel(), [2, 98])
    plt.imshow(data[shot].T,aspect='auto',cmap=cmap,vmin=d_vmin,vmax=d_vmax)
    plt.ylabel('Time (ms)', fontsize=14)
    plt.xlabel('Trace Number',fontsize=14)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.colorbar()
    plt.show()
