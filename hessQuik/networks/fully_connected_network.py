from hessQuik.networks import NN
from hessQuik.layers import singleLayer
import hessQuik.activations as act
from typing import Union, Tuple, List
from copy import deepcopy


class fullyConnectedNN(NN):
    r"""
    Fully-connected network where every layer is a single layer.  Let :math:`u_0 = x` be the input into the network.
    The construction is of the form

    .. math::

        \begin{align}
            u_1 &= \sigma(K_1 u_0 + b_1)\\
            u_2 &= \sigma(K_2 u_1 + b_2)\\
                &\vdots \\
            u_{\ell} &= \sigma(K_{\ell} u_{\ell-1} + b_{\ell})
        \end{align}

    where :math:`\ell` is the number of layers.
    Each vector of features :math:`u_i` is of size :math:`(n_s, n_i)` where :math:`n_s` is the number of samples
    and :math:`n_i` is the dimension or width of the hidden features on layer :math:`i`.
    Users choose the widths of the network and the activation function :math:`\sigma`.
    """

    def __init__(self, widths: Union[Tuple, List], act: act.hessQuikActivationFunction = act.softplusActivation(),
                 device=None, dtype=None):
        r"""
        :param widths: dimension of hidden features
        :type widths: tuple or list
        :param act: hessQuik activation function. Default: hessQuik.activations.softplusActivationFunction
        :type act: hessQuik.activations.hessQuikActivationFunction
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        args = ()
        for i in range(len(widths) - 1):
            args += (singleLayer(widths[i], widths[i + 1], act=deepcopy(act), **factory_kwargs),)

        super(fullyConnectedNN, self).__init__(*args)


if __name__ == '__main__':
    import torch
    from hessQuik.utils import input_derivative_check
    torch.set_default_dtype(torch.float64)

    # problem setup
    nex = 11
    d = 3
    x = torch.randn(nex, d)
    f = fullyConnectedNN([d, 2, 5, 1], act=act.softplusActivation())

    print('======= FORWARD =======')
    input_derivative_check(f, x, do_Hessian=True, verbose=True, forward_mode=True)

    print('======= BACKWARD =======')
    input_derivative_check(f, x, do_Hessian=True, verbose=True, forward_mode=False)

    # widths1 = [2, 3]
    # widths2 = [4, 5]
    # widths3 = [7, 6, 2]
    #
    # f = NN(fullyConnectedNN([d] + widths1, act=act.antiTanhActivation()),
    #        fullyConnectedNN([widths1[-1]] + widths2, act=act.identityActivation()),
    #        fullyConnectedNN([widths2[-1]] + widths3, act=act.softplusActivation()))
    #
    # print('======= FORWARD NN =======')
    # input_derivative_check(f, x, do_Hessian=True, verbose=True, forward_mode=True)
    #
    # print('======= BACKWARD NN =======')
    # input_derivative_check(f, x, do_Hessian=True, verbose=True, forward_mode=False)
