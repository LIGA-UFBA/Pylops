import numpy as np
from typing import Union
from copy import deepcopy
from wavelets import Ricker
from numpy.linalg import norm
from utils import (create_mask_value, Wiener_Filt, get_alfa_g)

class FWI_Multiscale():
    def __init__(self, operator, vp_init, dobs):

        """
        Full-Waveform Inversion (FWI) using a multiscale approach.
    
        This class performs multiscale Full-Waveform Inversion (FWI) by inverting seismic data at multiple frequency bands. The inversion starts with lower frequencies and progresses to higher frequencies to refine the velocity model. The method includes a gradient descent update for the velocity model based on the residuals between the observed and predicted seismic data.
    
        Attributes
        ----------
        op : object
            Forward operator used to model seismic data.
        vp : :obj:`numpy.ndarray`
            Initial velocity model, updated during the inversion process.
        dobs : :obj:`numpy.ndarray`
            Observed seismic data to match.
        fo : dict
            Dictionary storing the objective function values for each frequency.
    
        Methods
        -------
        run(f_targ, iterations, step, freqs, vp_min, vp_max, water_layer)
            Performs the FWI inversion by iterating over multiple frequencies and updating the velocity model.
        """
        
        self.op = operator
        self.vp = vp_init
        self.dobs = dobs
        self.fo = {} 

    def run(self, 
            ftarg: Union[float, int], 
            iterations: int, 
            step: int,  
            vp_min: float, 
            vp_max: float,
            water_layer: int, freqs: Union[list, tuple, np.ndarray] = None) -> tuple:

        """
        Perform Full-Waveform Inversion (FWI) using a multiscale approach.

        This method iterates through a range of frequencies, starting from a target frequency `f_targ` and progressing to higher frequencies. For each frequency, it performs the inversion using a gradient descent approach to update the velocity model.

        Parameters
        ----------
        f_targ : float or int
            Target frequency to start the inversion process.
        iterations : int
            Number of iterations for each frequency.
        step : int
            Step size for frequency range increment.
        freqs : list, tuple, or numpy.ndarray, optional
            Specific frequencies to use. If not provided, frequencies are generated based on `f_targ` and `step`.
        vp_min : float
            Minimum value for the velocity model (used for clipping).
        vp_max : float
            Maximum value for the velocity model (used for clipping).
        water_layer : int
            Number of layers near the water region that should not be updated during inversion.

        Returns
        -------
        tuple
            A tuple containing:
            - :obj:`numpy.ndarray`: The updated velocity model.
            - dict: The history of the objective function values for each frequency.
            - :obj:`numpy.ndarray`: The filtered observed data.
        
        Notes
        -----
        - The method assumes that the velocity model and data are in numpy array format.
        - The water layer is excluded from velocity updates by zeroing the gradients in that region.
        - The objective function (FO) is the sum of squared residuals between observed and predicted data.
        """
        
        fpeak = self.op.geometry.f0 * 1e3  
        orig_wav = self.op.geometry.src.wavelet  
        
        if freqs and not step:
            pass  
        elif step and not freqs:
            freqs = np.arange(ftarg, fpeak + step, step)  
        else:
            raise NotImplementedError('You must choose between using specific frequencies or setting a step for frequency calculation')
        
        for ifreq, freq in enumerate(freqs):
            print('==============================================================')
            print(f'Working with frequency: {freq} Hz -> {ifreq + 1}/{len(freqs)}')
            print('==============================================================')
            history = np.zeros(iterations)  

            if freq == fpeak:
                dobsfilt = self.dobs  
                self.op.updatesrc(orig_wav)  
            else:
                target_wav = Ricker(nt=self.op.geometry.nt, dt=self.op.geometry.dt * 1e-3, fpeak=freq)
                
                
                dobsfilt = Wiener_Filt(wav_orig=orig_wav, 
                                       wav_targ=target_wav, 
                                       orig_data=self.dobs)
                self.op.updatesrc(target_wav)  

            for iter in range(iterations):
                print("Iteration ", iter + 1)
                
                dcalc = self.op * self.vp

                res = dobsfilt - dcalc

                FO = 0.5 * norm(res)**2
                print("FO: ", FO)
                print('')
                
                grads = self.op.H * res 

                history[iter] = FO
                self.fo[freq] = history.tolist()

                grads[:, 0:water_layer] = 0. 

                if iter == 0:
                    alfa = 0.05 / np.max(grads)  
                else:
                    yk = grads - gradp  
                    sk = self.vp - vpp  
                    alfa = get_alfa_g(yk, sk)  

                gradp = deepcopy(grads)
                vpp = deepcopy(self.vp)
               
                self.vp = self.vp - alfa * grads
                self.vp[:, 0:water_layer] = self.vp[:, 0:water_layer]  
            
                np.putmask(self.vp, self.vp > vp_max, vp_max)
                np.putmask(self.vp, self.vp < vp_min, vp_min)

        print('FWI is finished!')
        return self.vp, self.fo, dobsfilt
