
from hessQuik.networks import NN
from hessQuik.layers import ICNNLayer
import hessQuik.activations as act
from typing import Union, Tuple, List


class ICNN(NN):

    def __init__(self, input_dim: int, widths: Union[Tuple, List], act: act.activationFunction = act.identityActivation(),
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(ICNN, self).__init__()

        for i, w in enumerate(range(1, len(widths))):
            self.add_module(str(i), ICNNLayer(input_dim, widths[i], widths[i + 1], act=act, **factory_kwargs))
