from hessQuik.networks import NN
from hessQuik.layers import resnetLayer
import hessQuik.activations as act
from copy import deepcopy


class resnetNN(NN):

    def __init__(self, width: int, depth: int, h: float = 1.0, act: act.activationFunction = act.identityActivation(),
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(resnetNN, self).__init__()

        for i in range(depth):
            self.add_module(str(i), resnetLayer(width, h=h, act=deepcopy(act), **factory_kwargs))


if __name__ == '__main__':
    import torch
    from hessQuik.utils import input_derivative_check
    torch.set_default_dtype(torch.float32)

    # problem setup
    nex = 11
    d = 3
    x = torch.randn(nex, d)

    f = resnetNN(d, 4, h=0.5, act=act.softplusActivation())

    print('======= FORWARD =======')
    input_derivative_check(f, x, do_Hessian=True, verbose=True, reverse_mode=False)

    print('======= BACKWARD =======')
    input_derivative_check(f, x, do_Hessian=True, verbose=True, reverse_mode=True)

