
from hessQuik.networks import NN
from hessQuik.layers import ICNNLayer
import hessQuik.activations as act
from typing import Union, Tuple, List
from copy import deepcopy


class ICNN(NN):

    def __init__(self, input_dim: int, widths: Union[Tuple, List], act: act.activationFunction = act.identityActivation(),
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(ICNN, self).__init__()

        for i, w in enumerate(range(1, len(widths))):
            self.add_module(str(i), ICNNLayer(input_dim, widths[i], widths[i + 1], act=deepcopy(act), **factory_kwargs))


if __name__ == '__main__':
    import torch
    from hessQuik.tests.utils import DerivativeCheckTestsNetwork
    torch.set_default_dtype(torch.float64)

    # problem setup
    nex = 11  # no. of examples
    d = 3  # no. of input features
    ms = [None, 5, 2, 7]  # no. of output features
    x = torch.randn(nex, d)
    dx = torch.randn_like(x)

    f = ICNN(d, ms, act=act.quadraticActivation())

    # forward tests
    derivativeTests = DerivativeCheckTestsNetwork()

    print('======= FORWARD =======')
    derivativeTests.run_forward_hessian_test(f, x, dx, verbose=True)

    print('======= BACKWARD =======')
    derivativeTests.run_backward_hessian_test(f, x, dx, verbose=True)
