from examples.seismic import SeismicModel
from devito import Constant

class AcousticModel(SeismicModel):
    _known_parameters = SeismicModel._known_parameters + ['gamma', 'vsh', 'sw']

    def _initialize_physics(self, vp, space_order, **kwargs):
        params = []
        rho = kwargs.get('rho')

        self.rho = self._gen_phys_param(rho, 'rho', space_order)
        self.vp = self._gen_phys_param(vp, 'vp', space_order)
    
        # Initialize rest of the input physical parameters
        for name in self._known_parameters:
            if kwargs.get(name) is not None:
                field = self._gen_phys_param(kwargs.get(name), name, space_order)
                setattr(self, name, field)
                params.append(name)

        self._initialize_constants(**kwargs)

    def _initialize_constants(self, **kwargs):
        _constants = ['a1', 'a2', 'a3', 'a4']
        # Initialize rest of the input physical parameters
        for name in _constants:
            if kwargs.get(name) is not None:
                const = Constant(name=name, value=kwargs.get(name))
                setattr(self, name, const)
                self._physical_parameters.update([name])