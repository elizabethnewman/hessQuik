
from hessQuik.networks import NN
from hessQuik.layers import ICNNLayer
import hessQuik.activations as act
from typing import Union, Tuple, List
from copy import deepcopy


class ICNN(NN):
    r"""
    Input Convex Neural Networks (ICNN) were proposed in the paper `Input Convex Neural Networks`_ by Amos, Xu, and Kolter.
    The network is constructed such that it is convex with respect to the network inputs :math:`x`.
    It is constructed via

    .. _Input Convex Neural Networks: https://arxiv.org/abs/1609.07152

    .. math::

        \begin{align}
            u_1 &= \sigma(K_1 x + b_1)\\
            u_2 &= \sigma(L_2^+ u_1 + K_2 x + b_2)\\
                &\vdots\\
            u_{\ell} &= \sigma(L_{\ell}^+ u_{\ell-1} + K_{\ell} x + b_{\ell})
        \end{align}

    where :math:`\ell` is the number of layers.
    Here, :math:`(\cdot)^+` is a function that forces the matrix to have only nonnegative entries.
    The activation function :math:`\sigma` must be convex and non-decreasing.
    """

    def __init__(self, input_dim: int, widths: Union[Tuple, List],
                 act: act.hessQuikActivationFunction = act.softplusActivation(),
                 device=None, dtype=None):
        r"""

        :param input_dim: dimension of input features
        :type input_dim: int
        :param widths: dimension of hidden features
        :type widths: tuple or list
        :param act: hessQuik activation function. Default: hessQuik.activations.softplusActivationFunction
        :type act: hessQuik.activations.hessQuikActivationFunction
        """
        factory_kwargs = {'device': device, 'dtype': dtype}

        args = ()
        for i, w in enumerate(range(1, len(widths))):
            args += (ICNNLayer(input_dim, widths[i], widths[i + 1], act=deepcopy(act), **factory_kwargs),)

        super(ICNN, self).__init__(*args)


if __name__ == '__main__':
    import torch
    from hessQuik.utils import input_derivative_check
    torch.set_default_dtype(torch.float64)

    # problem setup
    nex = 11  # no. of examples
    d = 3  # no. of input features
    ms = [None, 5, 2, 7]  # no. of output features
    x = torch.randn(nex, d)

    f = ICNN(d, ms, act=act.quadraticActivation())

    print('======= FORWARD =======')
    input_derivative_check(f, x, do_Hessian=True, verbose=True, forward_mode=True)

    print('======= BACKWARD =======')
    input_derivative_check(f, x, do_Hessian=True, verbose=True, forward_mode=False)
