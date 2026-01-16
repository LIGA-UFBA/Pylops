import sys
sys.path.append('../')

import numpy as np
from pathlib import Path 
from pylops.utils import *
from numpy.linalg import norm
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from devito import configuration
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter
from pylops.waveeqprocessing.twoway import PetroAcousticWave2D
from PetroFWI import PetroAcoustic_FWI_Multiscale

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
    'ns': 40,   'ds': 175,    'sz': 0.,
    'nr': 98,   'dr': 25,     'rz': 0.,
    'nt': 3000, 'dt': 0.002,   't0': 0.,
    'nw': 10,   'freq': 20
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
iterations_phase1 = 400

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

# ====== Função objetivo + gradiente (usa x) ======
def fun_and_grad_x(x):
    phi_curr, vsh_curr, sw_curr = unpack(x)
    params_init = np.stack([phi_curr, vsh_curr, sw_curr])

    dcalc = Dop * params_init
    res = dobs - dcalc

    FO = 0.5 * norm(res) ** 2

    grads_phi, grads_vsh, grads_sw = Dop.H * res
    # zera grad no water layer
    grads_phi[:, 0:par['nw']] = 0. 
    grads_vsh[:, 0:par['nw']] = 0.
    grads_sw[:, 0:par['nw']] = 0.

    grads_flat = pack(grads_phi, grads_vsh, grads_sw)
    return FO, grads_flat

# ====== Bounds / helpers ======
def freeze_bounds(arr0, vmin, vmax, nwater):
    """Limita arr0 entre [vmin, vmax] e congela water layer (0:nwater)."""
    lo = np.full_like(arr0, vmin, dtype=float)
    hi = np.full_like(arr0, vmax, dtype=float)
    if nwater > 0:
        lo[:, :nwater] = arr0[:, :nwater]
        hi[:, :nwater] = arr0[:, :nwater]
    return list(zip(lo.ravel(), hi.ravel()))

def freeze_all(arr0):
    """Congela TODO o campo (útil para travar um parâmetro em toda a malha)."""
    lo = arr0.astype(float).ravel()
    hi = arr0.astype(float).ravel()
    return list(zip(lo, hi))

b_phi_free   = freeze_bounds(init_phi, phi_min, phi_max, par['nw'])
b_vsh_free   = freeze_bounds(init_vsh, vsh_min, vsh_max, par['nw'])
b_sw_free    = freeze_bounds(init_sw,  sw_min,  sw_max,  par['nw'])

# ====== FASE 1: inverter apenas phi e vsh (sw totalmente congelado) ======
bounds_phase1 = b_phi_free + b_vsh_free + freeze_all(init_sw)
x0_phase1 = np.r_[init_phi.ravel(), init_vsh.ravel(), init_sw.ravel()]

history1 = []
with open("txt/fo_results_phase1_phi_vsh.txt", "w") as f:
    f.write("Iterations Results (Phase 1: phi+vsh only)\n")
    f.write("==========================================\n\n")

def cb1(xk):
    fk, _ = fun_and_grad_x(xk)
    history1.append(float(fk))
    with open("txt/fo_results_phase1_phi_vsh.txt", "a") as f:
        f.write(f"Iter: {len(history1)} | FO: {fk}\n")

res_phase1 = minimize(
    fun=lambda x: fun_and_grad_x(x)[0],
    x0=x0_phase1,
    jac=lambda x: fun_and_grad_x(x)[1],
    method='L-BFGS-B',
    bounds=bounds_phase1,
    callback=cb1,
    options=dict(
        maxiter=iterations_phase1,
        ftol=1e-12,
        maxcor=20,
        disp=True
    )
)

phi_p1, vsh_p1, sw_p1 = unpack(res_phase1.x)  # sw_p1 == init_sw (congelado)
# reforça limites + water layer
np.clip(phi_p1, phi_min, phi_max, out=phi_p1)
np.clip(vsh_p1, vsh_min, vsh_max, out=vsh_p1)
phi_p1[:, 0:par['nw']] = phi[:, 0:par['nw']]
vsh_p1[:, 0:par['nw']] = vsh[:, 0:par['nw']]
sw_p1[:,  0:par['nw']] = sw[:,  0:par['nw']]

# ====== FASE 2: inverter apenas sw (phi e vsh congelados nos resultados da fase 1) ======
b_phi_fixed = freeze_all(phi_p1)
b_vsh_fixed = freeze_all(vsh_p1)
b_sw_free2  = freeze_bounds(init_sw, sw_min, sw_max, par['nw'])  # mantém travada water layer

bounds_phase2 = b_phi_fixed + b_vsh_fixed + b_sw_free2
x0_phase2 = np.r_[phi_p1.ravel(), vsh_p1.ravel(), init_sw.ravel()]
iterations_phase2=400

history2 = []
with open("txt/fo_results_phase2_sw.txt", "w") as f:
    f.write("Iterations Results (Phase 2: sw only)\n")
    f.write("=====================================\n\n")

def cb2(xk):
    fk, _ = fun_and_grad_x(xk)
    history2.append(float(fk))
    with open("txt/fo_results_phase2_sw.txt", "a") as f:
        f.write(f"Iter: {len(history2)} | FO: {fk}\n")

res_phase2 = minimize(
    fun=lambda x: fun_and_grad_x(x)[0],
    x0=x0_phase2,
    jac=lambda x: fun_and_grad_x(x)[1],
    method='L-BFGS-B',
    bounds=bounds_phase2,
    callback=cb2,
    options=dict(
        maxiter=iterations_phase2,
        ftol=1e-12,
        maxcor=20,
        disp=True
    )
)

# ====== Resultados finais ======
phi_inv, vsh_inv, sw_inv = unpack(res_phase2.x)

# garantir limites e travar water layer
np.clip(phi_inv, phi_min, phi_max, out=phi_inv)
np.clip(vsh_inv, vsh_min, vsh_max, out=vsh_inv)
np.clip(sw_inv,  sw_min,  sw_max,  out=sw_inv)
phi_inv[:, 0:par['nw']] = phi[:, 0:par['nw']]
vsh_inv[:, 0:par['nw']] = vsh[:, 0:par['nw']]
sw_inv[:,  0:par['nw']] = sw[:,  0:par['nw']]

global_history = [history1, history2]

# salvar em NPZ comprimido
np.savez_compressed("results/petro_inv_phases_conventional.npz", phi=phi_inv, vsh=vsh_inv, sw=sw_inv, fo=global_history)

plt.figure(figsize=(10, 6))
plt.plot(history1, color='black')
plt.plot(history2, color='red')

# Títulos e rótulos
plt.xlabel('Iterations', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.savefig('images/FO_FWI_phases_conventional.png', format='png', dpi=200)

# ----- Plots (mesma lógica do seu script) -----
# Convergência
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
plt.savefig('images/inv_phi_multiscale_minimize/results_FWI_phases_Conventional.png', format='png', dpi=200)