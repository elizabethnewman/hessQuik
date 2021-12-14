from hessQuik.networks import NN
from hessQuik.layers import singleLayer
import hessQuik.activations as act
from typing import Union, Tuple, List
from copy import deepcopy


class fullyConnectedNN(NN):

    def __init__(self, widths: Union[Tuple, List], act: act.hessQuikActivationFunction = act.identityActivation(),
                 device=None, dtype=None, **kwargs):
        factory_kwargs = {'device': device, 'dtype': dtype}
        args = ()
        for i in range(len(widths) - 1):
            args += (singleLayer(widths[i], widths[i + 1], act=deepcopy(act), **factory_kwargs),)

        super(fullyConnectedNN, self).__init__(*args, **kwargs)


if __name__ == '__main__':
    import torch
    from hessQuik.utils import input_derivative_check
    torch.set_default_dtype(torch.float64)

    # problem setup
    nex = 11
    d = 3
    x = torch.randn(nex, d)

    # print('======= FORWARD =======')
    # f = fullyConnectedNN([d, 2, 5], act=act.softplusActivation(), reverse_mode=False)
    # input_derivative_check(f, x, do_Hessian=True, verbose=True)
    #
    # print('======= BACKWARD =======')
    # f = fullyConnectedNN([d, 2, 5, 1], act=act.softplusActivation())
    # input_derivative_check(f, x, do_Hessian=True, verbose=True)

    widths1 = [2, 3]
    widths2 = [4, 5]
    widths3 = [7, 6, 2]

    f = NN(fullyConnectedNN([d] + widths1, act=act.antiTanhActivation()),
           fullyConnectedNN([widths1[-1]] + widths2, act=act.identityActivation()),
           fullyConnectedNN([widths2[-1]] + widths3, act=act.softplusActivation()))
    input_derivative_check(f, x, do_Hessian=True, verbose=True)
