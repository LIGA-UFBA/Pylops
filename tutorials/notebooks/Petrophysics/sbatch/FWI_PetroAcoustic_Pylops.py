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
# from utils import (plot_data, Frequency_spectrum, plot_freq_spectrum, create_mask_value, Wiener_Filt)

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
    'ns': 100,  'ds': 175,    'sz': 0.,
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
init_vp = gaussian_filter(vp, sigma=(5,5))

init_phi[:,0:par['nw']] = phi[:,0:par['nw']]
init_vsh[:,0:par['nw']] = vsh[:,0:par['nw']]
init_sw[:,0:par['nw']] = sw[:,0:par['nw']]
init_vp[:,0:par['nw']] = vp[:,0:par['nw']]

Dop = PetroAcousticWave2D(shape=shape,origin=origin, spacing=spacing, vp=vp*1e3, phi=phi, vsh=vsh, sw=sw, nbl=nbl, 
                          space_order=space_order,src_x=x_s[:,0], src_z=x_s[:,1], rec_x=x_r[:,0],
                          rec_z=x_r[:,1], t0=par['t0'], tn=par['nt'], src_type=src_type, f0=par['freq'],
                          dtype=dtype,op_name="fwd", dt=par['dt']*1e3, multi_input=True, **coef)

params = np.stack([vp, phi, vsh, sw])  
dobs = Dop * params
min_max_list = [(phi_min,phi_max), (vsh_min, vsh_max), (sw_min, sw_max)]

fwi = PetroAcoustic_FWI_Multiscale(operator=Dop,
                           vp_init=init_vp,
                           phi_init=init_phi,
                           vsh_init = init_vsh,
                           sw_init = init_sw,
                           dobs=dobs)

phi_inv, vsh_inv, sw_inv, fo, dobsfilt = fwi.run(ftarg=2, iterations=30, step=2, witer=10,
                                                 min_max=min_max_list, water_layer=par['nw'],freqs=None)

extent = [0, nx*spacing[0], nz*spacing[1], 0]
plt.figure(figsize=(12,10))
plt.subplot(3,3,1)
plt.imshow(init_phi.T, cmap='jet',extent=extent)
plt.title('Initial Porosity Model', fontsize=12)
plt.colorbar()
plt.subplot(3,3,2)
plt.imshow(phi.T, cmap='jet',extent=extent)
plt.title('True Porosity Model', fontsize=12)
plt.colorbar()
plt.subplot(3,3,3)
plt.imshow(phi_inv.T, cmap='jet',extent=extent)
plt.title('Inverted Porosity Model', fontsize=12)
plt.colorbar()
plt.subplot(3,3,4)
plt.imshow(init_vsh.T, cmap='jet',extent=extent)
plt.title('Initial Shale Model', fontsize=12)
plt.colorbar()
plt.subplot(3,3,5)
plt.imshow(vsh.T, cmap='jet',extent=extent)
plt.title('True Shale Model', fontsize=12)
plt.colorbar()
plt.subplot(3,3,6)
plt.imshow(vsh_inv.T, cmap='jet',extent=extent)
plt.title('Inverted Shale Model', fontsize=12)
plt.colorbar()
plt.subplot(3,3,7)
plt.imshow(init_sw.T, cmap='jet',extent=extent)
plt.title('Initial Water Sat. Model', fontsize=12)
plt.colorbar()
plt.subplot(3,3,8)
plt.imshow(sw.T, cmap='jet',extent=extent)
plt.title('True Water Sat. Model', fontsize=12)
plt.colorbar()
plt.subplot(3,3,9)
plt.imshow(sw_inv.T, cmap='jet',extent=extent)
plt.title('Inverted Water Sat. Model', fontsize=12)
plt.colorbar()
plt.tight_layout()
plt.savefig('images/Results_PylopsPetroAcousitc_FWI.png', format='png', dpi=200)


plt.figure(figsize=(10, 6))
plt.plot(fo)
plt.xlabel('Iterations')
plt.ylabel('Amplitude')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('images/FO_PylopsPetroAcousitc_FWI.png', format='png', dpi=200)
plt.show()