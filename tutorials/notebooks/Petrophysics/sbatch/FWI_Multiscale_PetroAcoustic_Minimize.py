import sys
sys.path.append('../')

import numpy as np
from pathlib import Path 
from pylops.utils import *
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from devito import configuration
from scipy.ndimage import gaussian_filter
from pylops.waveeqprocessing.twoway import PetroAcousticWave2D
from PetroFWI import PetroAcoustic_FWI_Multiscale
from numpy.linalg import norm
from utils import Wiener_Filt
from scipy.optimize import minimize
from wavelets import Ricker

configuration['log-level'] = 'ERROR'
from seis2rock.pem_seis2rock import pem_seis2rock


BASE_DIR = Path.cwd()
path = BASE_DIR.parent.parent / "data" / "smeaheia_petrophysics.npz"

phi_orig, vsh_orig, sw_orig, depth = np.load(path)['phi'], np.load(path)['vsh'], np.load(path)['sw'], np.load(path)['depth']
xaxis, nx, nz, dz = np.load(path)['xaxis'], np.load(path)['nx'], np.load(path)['nz'], np.load(path)['dz']

dx = xaxis[1] - xaxis[0]
dz = depth[1] - depth[0]

# Novo espaçamento
new_spacing = (25, 25)

# Fatores de escala (novo tamanho / antigo tamanho)
zoom_factors = (dx / new_spacing[0], dz / new_spacing[1])

# Redimensionar com interpolação "nearest"
phi = zoom(phi_orig, zoom_factors, order=0)
vsh = zoom(vsh_orig, zoom_factors, order=0)
sw = zoom(sw_orig, zoom_factors, order=0)

phi_min, phi_max = np.min(phi), np.max(phi)
vsh_min, vsh_max = np.min(vsh), np.max(vsh)
sw_min, sw_max = np.min(sw), np.max(sw)

# Add water column
water_lenght = 0                                # Water column thickness (m)
nwater = int(water_lenght / new_spacing[1])     # Number of samples of water column
nwater = 10

vp, vs, rho = pem_seis2rock(phi=phi, vsh=vsh, sw=sw)      # Calculate elastic properties using Seis2Rock (Gomes et al., 2024)

vp /= 1000      # Convert from m/s to km/s

# Modelling parameters (in meters, seconds and Hz)
par = {
    'nx': 98,   'dx': 25,     'ox': 0.,
    'nz': 100,  'dz': 25,     'oz': 0.,
    'ns': 40,  'ds': 175,    'sz': 0.,
    'nr': 98,   'dr': 25,     'rz': 0.,
    'nt': 3000, 'dt': 0.002,   't0': 0.,
    'nw': 5,   'freq': 20
}

coef = {'a1': -0.43, 'a2': 0.88, 'a3': 0.57, 'a4': 1.72}

shape = (par['nx'], par['nz'])
spacing = (par['dx'], par['dz'])
origin = (par['ox'], par['oz'])
src_type = 'Ricker' # or 'Ricker_Devito' if you want to use Devito Source
dtype = np.float32
space_order = 8
nbl = 50

# Setting receivers positions
x_r = np.zeros((par['nr'], 2))
x_r[:, 0] = np.arange(par['nr']) * par['dr'] # receivers positions
x_r[:, 1] = par['rz']

# Setting source positions
model_domain_size = ((shape[0]-1)*spacing[0], (shape[1]-1)*spacing[1])
x_s = np.zeros((par['ns'], 2))
x_s[:,0] = np.linspace(0., model_domain_size[0], num=par['ns'])
x_s[:,1] = origin[1] + spacing[1] * 2

init_phi = gaussian_filter(phi, sigma=(5,5))
init_vsh = gaussian_filter(vsh, sigma=(5,5))
init_sw = gaussian_filter(sw, sigma=(5,5))
# init_vp = gaussian_filter(vp, sigma=(5,5))

init_phi[:,0:par['nw']] = phi[:,0:par['nw']]
init_vsh[:,0:par['nw']] = vsh[:,0:par['nw']]
init_sw[:,0:par['nw']] = sw[:,0:par['nw']]
# init_vp[:,0:par['nw']] = vp[:,0:par['nw']]

Dop = PetroAcousticWave2D(shape=shape,origin=origin, spacing=spacing, vp=vp*1e3, phi=phi, vsh=vsh, sw=sw, nbl=nbl, 
                          space_order=space_order,src_x=x_s[:,0], src_z=x_s[:,1], rec_x=x_r[:,0],
                          rec_z=x_r[:,1], t0=par['t0'], tn=par['nt'], src_type=src_type, f0=par['freq'],
                          dtype=dtype,op_name="fwd", dt=par['dt']*1e3, multi_input=True, **coef)

params = np.stack([phi, vsh, sw])  
dobs = Dop * params
min_max_list = [(phi_min,phi_max), (vsh_min, vsh_max), (sw_min, sw_max)]
iterations=200

# ----- Objective (FO) e Gradiente -----
shp = phi.shape  # (nx, nz)

def pack(phiA, vshA, swA):
    return np.r_[phiA.ravel(), vshA.ravel(), swA.ravel()]

def unpack(x):
    n = shp[0]*shp[1]
    phiA = x[0:n].reshape(shp)
    vshA = x[n:2*n].reshape(shp)
    swA  = x[2*n:3*n].reshape(shp)
    return phiA, vshA, swA

# ====== Bounds para phi/vsh/sw + congelar water layer ======
def freeze_bounds(arr0, vmin, vmax, nwater):
    lo = np.full_like(arr0, vmin, dtype=float)
    hi = np.full_like(arr0, vmax, dtype=float)
    if nwater > 0:
        lo[:, :nwater] = arr0[:, :nwater]
        hi[:, :nwater] = arr0[:, :nwater]
    return list(zip(lo.ravel(), hi.ravel()))

b_phi = freeze_bounds(init_phi, phi_min, phi_max, par['nw'])
b_vsh = freeze_bounds(init_vsh, vsh_min, vsh_max, par['nw'])
b_sw  = freeze_bounds(init_sw,  sw_min,  sw_max,  par['nw'])
bounds = b_phi + b_vsh + b_sw

# ====== x0 inicial (phi, vsh, sw) ======
x0 = pack(init_phi, init_vsh, init_sw)

# ====== Histórico e log ======
history = []
with open("txt/fo_results_FWI_multiscale.txt", "w") as f:
    f.write("Iterations Results \n")
    f.write("========================\n\n")

fmin, fmax, fstep = 2, par['freq'], 2
freqs = np.arange(fmin, fmax + fstep, fstep)
print(f"Frequências de inversão multiescala: {freqs} Hz")

# Armazena wavelet original (para filtragem)
orig_wav = Dop.geometry.src.wavelet
fpeak = Dop.geometry.f0 * 1e3  # Hz

# Modelos iniciais (para primeira frequência)
phi_curr = init_phi.copy()
vsh_curr = init_vsh.copy()
sw_curr  = init_sw.copy()

# Histórico global
global_history = []

for ifreq, freq in enumerate(freqs):
    print("\n" + "="*70)
    print(f"  Etapa {ifreq+1}/{len(freqs)} - Frequência alvo: {freq} Hz")
    print("="*70)

    # --- Filtragem do dado observado ---
    if np.isclose(freq, fpeak):
        # frequência máxima: usa dados originais
        dobs_filt = dobs
        Dop.updatesrc(orig_wav)
    else:
        # Cria wavelet alvo (Ricker) e aplica Wiener Filter
        target_wav = Ricker(nt=Dop.geometry.nt, dt=Dop.geometry.dt * 1e-3, fpeak=freq)
        dobs_filt = Wiener_Filt(wav_orig=orig_wav, wav_targ=target_wav, orig_data=dobs)
        Dop.updatesrc(target_wav)

    # --- Atualiza modelo inicial e bounds ---
    x0 = pack(phi_curr, vsh_curr, sw_curr)
    b_phi = freeze_bounds(phi_curr, phi_min, phi_max, par['nw'])
    b_vsh = freeze_bounds(vsh_curr, vsh_min, vsh_max, par['nw'])
    b_sw  = freeze_bounds(sw_curr,  sw_min,  sw_max,  par['nw'])
    bounds = b_phi + b_vsh + b_sw

    # --- Define função objetivo com dados filtrados desta frequência ---
    def fun_and_grad_multiscale(x):
        phi_tmp, vsh_tmp, sw_tmp = unpack(x)
        params_tmp = np.stack([phi_tmp, vsh_tmp, sw_tmp])

        dcalc = Dop * params_tmp
        res = dobs_filt - dcalc

        FO = 0.5 * norm(res)**2
        grads_phi, grads_vsh, grads_sw = Dop.H * res

        # zera gradientes na water layer
        grads_phi[:, 0:par['nw']] = 0.
        grads_vsh[:, 0:par['nw']] = 0.
        grads_sw[:,  0:par['nw']] = 0.

        grads_flat = pack(grads_phi, grads_vsh, grads_sw)
        return FO, grads_flat

    # --- Callback para logar o histórico ---
    history_freq = []
    def cb(xk):
        fk, _ = fun_and_grad_multiscale(xk)
        history_freq.append(float(fk))
        print(f"Iter {len(history_freq)} | FO: {fk:.3e}", end='\r')
        with open("txt/fo_results_FWI_multiscale.txt", "a") as f:
            f.write(f"Iter {len(history_freq)} | FO: {fk:.3e}\r")

    # --- Minimização ---
    res_opt = minimize(
        fun=lambda x: fun_and_grad_multiscale(x)[0],
        x0=x0,
        jac=lambda x: fun_and_grad_multiscale(x)[1],
        method='L-BFGS-B',
        bounds=bounds,
        callback=cb,
        options=dict(maxiter=iterations, ftol=1e-10, maxcor=20, disp=False)
    )

    # print(f"\n--> Etapa {ifreq+1} finalizada | FO final = {history_freq[-1]:.3e}")
    print('')
    global_history.append(history_freq)

    # --- Atualiza modelos para próxima etapa ---
    phi_curr, vsh_curr, sw_curr = unpack(res_opt.x)

    # Clipping e travamento da camada d'água
    np.clip(phi_curr, phi_min, phi_max, out=phi_curr)
    np.clip(vsh_curr, vsh_min, vsh_max, out=vsh_curr)
    np.clip(sw_curr,  sw_min,  sw_max,  out=sw_curr)
    phi_curr[:, 0:par['nw']] = phi[:, 0:par['nw']]
    vsh_curr[:, 0:par['nw']] = vsh[:, 0:par['nw']]
    sw_curr[:,  0:par['nw']] = sw[:,  0:par['nw']]

# --- Resultados finais após todas as frequências ---
phi_inv, vsh_inv, sw_inv = phi_curr, vsh_curr, sw_curr
# fo = np.stack(global_history, axis=0)

# Salva resultados
np.savez_compressed("results/petro_inv_multiscale.npz", phi=phi_inv, vsh=vsh_inv, sw=sw_inv)

# --- Plots finais ---
extent = [0, nx*spacing[0], nz*spacing[1], 0]
plt.figure(figsize=(12,10))
plt.subplot(3,3,1)
plt.imshow(init_phi.T, cmap='viridis',extent=extent)
plt.title('Initial Porosity Model', fontsize=12)
plt.colorbar()
plt.subplot(3,3,2)
plt.imshow(phi.T, cmap='viridis',extent=extent)
plt.title('True Porosity Model', fontsize=12)
plt.colorbar()
plt.subplot(3,3,3)
plt.imshow(phi_inv.T, cmap='viridis',extent=extent)
plt.title('Inverted Porosity Model', fontsize=12)
plt.colorbar()
plt.subplot(3,3,4)
plt.imshow(init_vsh.T, cmap='viridis',extent=extent)
plt.title('Initial Shale Model', fontsize=12)
plt.colorbar()
plt.subplot(3,3,5)
plt.imshow(vsh.T, cmap='viridis',extent=extent)
plt.title('True Shale Model', fontsize=12)
plt.colorbar()
plt.subplot(3,3,6)
plt.imshow(vsh_inv.T, cmap='viridis',extent=extent)
plt.title('Inverted Shale Model', fontsize=12)
plt.colorbar()
plt.subplot(3,3,7)
plt.imshow(init_sw.T, cmap='viridis',extent=extent)
plt.title('Initial Water Sat. Model', fontsize=12)
plt.colorbar()
plt.subplot(3,3,8)
plt.imshow(sw.T, cmap='viridis',extent=extent)
plt.title('True Water Sat. Model', fontsize=12)
plt.colorbar()
plt.subplot(3,3,9)
plt.imshow(sw_inv.T, cmap='viridis',extent=extent)
plt.title('Inverted Water Sat. Model', fontsize=12)
plt.colorbar()
plt.tight_layout()
plt.savefig('images/multiscale_minimize/results_FWI_Multiscale.png', format='png',dpi=200)

plt.figure(figsize=(10,6))
for i, hist in enumerate(global_history):
    plt.plot(hist / np.max(hist), label=f"{freqs[i]:.1f} Hz")  # Normaliza para melhor comparação

plt.xlabel("Iterations", fontsize=12)
plt.ylabel("Amplitude", fontsize=12)
plt.legend(title="Frequency (Hz)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("images/multiscale_minimize/FO_FWI_multiscale.png", dpi=200)