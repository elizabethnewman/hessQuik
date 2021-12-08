from hessQuik.networks import NN
from hessQuik.layers import singleLayer
import hessQuik.activations as act
from typing import Union, Tuple, List
from copy import deepcopy


class fullyConnectedNN(NN):

    def __init__(self, widths: Union[Tuple, List], act: act.activationFunction = act.identityActivation(),
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(fullyConnectedNN, self).__init__()

        for i, w in enumerate(range(len(widths) - 1)):
            self.add_module(str(i), singleLayer(widths[i], widths[i + 1], act=deepcopy(act), **factory_kwargs))


if __name__ == '__main__':
    import torch
    from hessQuik.tests.utils import DerivativeCheckTestsNetwork
    torch.set_default_dtype(torch.float64)

    # problem setup
    nex = 11
    d = 3
    x = torch.randn(nex, d)
    dx = torch.randn_like(x)

    f = fullyConnectedNN([d, 2, 5], act=act.softplusActivation())

    # forward tests
    derivativeTests = DerivativeCheckTestsNetwork()

    print('======= FORWARD =======')
    derivativeTests.run_forward_hessian_test(f, x, dx, verbose=True)

    print('======= BACKWARD =======')
    derivativeTests.run_backward_hessian_test(f, x, dx, verbose=True)
