import numpy as np
from typing import Union
from copy import deepcopy
from wavelets import Ricker
from numpy.linalg import norm
from utils import (create_mask_value, Wiener_Filt)
from utils_petro import (get_alfa_g, get_alfa_conv)

class PetroAcoustic_FWI_Multiscale():
    def __init__(self, operator, vp_init, phi_init, vsh_init, sw_init, dobs):

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
        self.phi = phi_init
        self.vsh = vsh_init
        self.sw = sw_init
        self.dobs = dobs
        self.fo = {} 

    def run(self, 
            ftarg: Union[float, int], 
            iterations: int, 
            witer: int,
            step: int,  
            min_max: list,
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

        phi_min, phi_max = min_max[0][0], min_max[0][1]
        vsh_min, vsh_max = min_max[1][0], min_max[1][1]
        sw_min, sw_max = min_max[2][0], min_max[2][1]
        
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
 
                params_init = np.stack([self.phi, self.vsh, self.sw])
                
                dcalc = self.op * params_init

                res = dobsfilt - dcalc

                FO = 0.5 * norm(res)**2
                print("FO: ", FO)
                print('')
                
                grads_phi, grads_vsh, grads_sw = self.op.H * res 

                history[iter] = FO
                self.fo[freq] = history.tolist()

                grads_phi[:, 0:water_layer] = 0. 
                grads_vsh[:, 0:water_layer] = 0.
                grads_sw[:, 0:water_layer] = 0.
            
                if iter == 0:
                    alfa = get_alfa_conv(grads_phi, grads_vsh, grads_sw)    
                else:
                    ykphi, skphi = grads_phi - gradp_phi, self.phi - pphi
                    ykvsh, skvsh = grads_vsh - gradp_vsh, self.vsh - pvsh
                    yksw, sksw = grads_sw - gradp_sw, self.sw - psw
                    
                    alfa = get_alfa_g(ykphi, ykvsh, yksw, skphi, skvsh, sksw)

                gradp_phi = deepcopy(grads_phi)
                pphi = deepcopy(self.phi)

                gradp_vsh = deepcopy(grads_vsh)
                pvsh = deepcopy(self.vsh)

                gradp_sw = deepcopy(grads_sw)
                psw = deepcopy(self.sw)
               
                self.phi = self.phi - alfa * grads_phi
                self.phi[:, 0:water_layer] = self.phi[:, 0:water_layer]  
                
                self.vsh = self.vsh - alfa * grads_vsh
                self.vsh[:, 0:water_layer] = self.vsh[:, 0:water_layer]
                
                self.sw = self.sw - alfa * grads_sw
                self.sw[:, 0:water_layer] = self.sw[:, 0:water_layer]  
            
                np.putmask(self.phi, self.phi > phi_max, phi_max)
                np.putmask(self.phi, self.phi < phi_min, phi_min)

                np.putmask(self.vsh, self.vsh > vsh_max, vsh_max)
                np.putmask(self.vsh, self.vsh < vsh_min, vsh_min)

                np.putmask(self.sw, self.sw > sw_max, sw_max)
                np.putmask(self.sw, self.sw < sw_min, sw_min)

                if iter > 0 and (history[iter] > history[iter-1]):
                    print(f'Wolfe Condition must be applied, FO = {history[iter]}')
                    for w in range(witer):
                        print(f"Global iteration {iter + 1}, LS iteration {w + 1}")
                        alfa = .5 * alfa
                        self.phi, self.vsh, self.sw = pphi, pvsh, psw
                        
                        self.phi = self.phi - alfa * grads_phi
                        self.vsh = self.vsh - alfa * grads_vsh
                        self.sw = self.sw - alfa * grads_sw
        
                        if water_layer > 0:
                            nbl = self.op.model.nbl
                        
                            phi0 = self.op.model.phi.data[nbl:-nbl, nbl:-nbl] #tirando a borda
                            vsh0 = self.op.model.vsh.data[nbl:-nbl, nbl:-nbl]
                            sw0  = self.op.model.sw.data[nbl:-nbl,  nbl:-nbl]
     
                            self.phi[:,0:water_layer] = phi0[:,0:water_layer] 
                            self.vsh[:,0:water_layer] = vsh0[:,0:water_layer] 
                            self.sw[:,0:water_layer] = sw0[:,0:water_layer] 
        
                        params_wolfe = np.stack([self.phi, self.vsh, self.sw]) 

                        if freq == fpeak:
                            self.op.updatesrc(orig_wav)
                        else:
                            self.op.updatesrc(target_wav)
                        
                        dcalc_wolfe = self.op * params_wolfe
                        res_wolfe = dobsfilt - dcalc_wolfe
                        FO_wolfe = 0.5 * norm(res_wolfe)**2
                        print("FO Wolfe: ", FO_wolfe)
                        print('')
        
                        if FO_wolfe < history[iter-1]:
                            print('Get out from line search - Wolfe conditions met')
                            break
                            
                        elif w == witer-1:
                            alfa = get_alfa_conv(grads_phi, grads_vsh, grads_sw)
                            self.phi = self.phi - alfa * grads_phi
                            self.vsh = self.vsh - alfa * grads_vsh
                            self.sw = self.sw - alfa * grads_sw
         
                            params_wolfe = np.stack([self.phi, self.vsh, self.sw])  
                            
                            if freq == fpeak:
                                self.op.updatesrc(orig_wav)
                            else:
                                self.op.updatesrc(target_wav)
                            
                            dcalc_wolfe = self.op * params_wolfe
                            FO_wolfe = 0.5 * norm(res)**2
                            print("FO Wolfe: ", FO_wolfe)
                            print('')
            
                            if FO_wolfe < history[iter-1]:
                                history[iter] = FO_wolfe
                    

        print('Petro FWI is finished!')
        return self.phi, self.vsh, self.sw, self.fo, dobsfilt


class PetroAcoustic_FWI():
    def __init__(self, operator, vp_init, phi_init, vsh_init, sw_init, dobs):

        """
        Petrophysics Acoustic Full-Waveform Inversion (FWI)
    
        This class performs a Petrophysics Acoustic Full-Waveform Inversion (FWI). The method includes a gradient descent update for the velocity model based on the residuals between the observed and predicted seismic data.
    
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
        run(f_targ, iterations, min_max, water_layer)
            Performs the FWI inversion by iterating over multiple frequencies and updating the velocity model.
        """
        
        self.op = operator
        self.vp = vp_init
        self.phi = phi_init
        self.vsh = vsh_init
        self.sw = sw_init
        self.dobs = dobs
        self.fo = {} 

    def run(self, 
            iterations: int,  
            witer: int,
            min_max: list,
            water_layer: int, freqs: Union[list, tuple, np.ndarray] = None) -> tuple:

        """
        Perform a Petrophysics Acoustic Full-Waveform Inversion (FWI).

        This method iterates through a range of iterations and performs the inversion using a gradient descent approach to update the velocity model.

        Parameters
        ----------
        iterations : int
            Number of iterations for each frequency.
        min_max : list
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

        phi_min, phi_max = min_max[0][0], min_max[0][1]
        vsh_min, vsh_max = min_max[1][0], min_max[1][1]
        sw_min, sw_max = min_max[2][0], min_max[2][1]
        history = np.zeros(iterations) 
          
        for iter in range(iterations):
            print("Iteration ", iter + 1)

            params_init = np.stack([self.phi, self.vsh, self.sw])  
            
            dcalc = self.op * params_init
            res = self.dobs - dcalc

            FO = 0.5 * norm(res)**2
            print("FO: ", FO)
            print('')
            
            grads_phi, grads_vsh, grads_sw = self.op.H * res 

            history[iter] = FO
        
            grads_phi[:, 0:water_layer] = 0. 
            grads_vsh[:, 0:water_layer] = 0.
            grads_sw[:, 0:water_layer] = 0.

            if iter == 0:
                alfa = get_alfa_conv(grads_phi, grads_vsh, grads_sw)
            else:
                ykphi, skphi = grads_phi - gradp_phi, self.phi - pphi
                ykvsh, skvsh = grads_vsh - gradp_vsh, self.vsh - pvsh
                yksw, sksw = grads_sw - gradp_sw, self.sw - psw
                
                alfa = get_alfa_g(ykphi, ykvsh, yksw, skphi, skvsh, sksw)
    
            gradp_phi = deepcopy(grads_phi)
            pphi = deepcopy(self.phi)

            gradp_vsh = deepcopy(grads_vsh)
            pvsh = deepcopy(self.vsh)

            gradp_sw = deepcopy(grads_sw)
            psw = deepcopy(self.sw)
           
            self.phi = self.phi - alfa * grads_phi
            self.phi[:, 0:water_layer] = self.phi[:, 0:water_layer]  
            
            self.vsh = self.vsh - alfa * grads_vsh
            self.vsh[:, 0:water_layer] = self.vsh[:, 0:water_layer]
            
            self.sw = self.sw - alfa * grads_sw
            self.sw[:, 0:water_layer] = self.sw[:, 0:water_layer]  
        
            np.putmask(self.phi, self.phi > phi_max, phi_max)
            np.putmask(self.phi, self.phi < phi_min, phi_min)

            np.putmask(self.vsh, self.vsh > vsh_max, vsh_max)
            np.putmask(self.vsh, self.vsh < vsh_min, vsh_min)

            np.putmask(self.sw, self.sw > sw_max, sw_max)
            np.putmask(self.sw, self.sw < sw_min, sw_min)

            if iter > 0 and (history[iter] > history[iter-1]):
                print(f'Wolfe Condition must be applied, FO = {history[iter]}')
                for w in range(witer):
                    print(f"Global iteration {iter + 1}, LS iteration {w + 1}")
                    alfa = .5 * alfa
                    self.phi, self.vsh, self.sw = pphi, pvsh, psw
                    
                    self.phi = self.phi - alfa * grads_phi
                    self.vsh = self.vsh - alfa * grads_vsh
                    self.sw = self.sw - alfa * grads_sw
    
                    if water_layer > 0:
                        nbl = self.op.model.nbl
                        
                        phi0 = self.op.model.phi.data[nbl:-nbl, nbl:-nbl] #tirando a borda
                        vsh0 = self.op.model.vsh.data[nbl:-nbl, nbl:-nbl]
                        sw0  = self.op.model.sw.data[nbl:-nbl,  nbl:-nbl]
 
                        self.phi[:,0:water_layer] = phi0[:,0:water_layer] 
                        self.vsh[:,0:water_layer] = vsh0[:,0:water_layer] 
                        self.sw[:,0:water_layer] = sw0[:,0:water_layer] 
    
                    params_wolfe = np.stack([self.phi, self.vsh, self.sw])  
                    
                    dcalc_wolfe = self.op * params_wolfe
                    res_wolfe = self.dobs - dcalc_wolfe
                    FO_wolfe = 0.5 * norm(res_wolfe)**2
                    print("FO Wolfe: ", FO_wolfe)
                    print('')
        
                    if FO_wolfe < history[iter-1]:
                        print('Get out from line search - Wolfe conditions met')
                        break
                        
                    elif w == witer-1:
                        alfa = get_alfa_conv(grads_phi, grads_vsh, grads_sw)
                        self.phi = self.phi - alfa * grads_phi
                        self.vsh = self.vsh - alfa * grads_vsh
                        self.sw = self.sw - alfa * grads_sw
    
                        params_wolfe = np.stack([self.phi, self.vsh, self.sw])  
                        
                        dcalc_wolfe = self.op * params_wolfe
                        FO_wolfe = 0.5 * norm(res)**2
                        print("FO Wolfe: ", FO_wolfe)
                        print('')
        
                        if FO_wolfe < history[iter-1]:
                            history[iter] = FO_wolfe
                    

        print('Petro FWI is finished!')
        return self.phi, self.vsh, self.sw, self.fo, dobsfilt