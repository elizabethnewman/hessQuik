from hessQuik.networks import NN
from hessQuik.layers import resnetLayer
import hessQuik.activations as act


class residualNN(NN):

    def __init__(self, width: int, depth: int, h: float, act: act.activationFunction = act.identityActivation(),
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(residualNN, self).__init__()

        for i in range(depth):
            self.add_module(str(i), resnetLayer(width, h=h, act=act, **factory_kwargs))




