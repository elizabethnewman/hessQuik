from hessQuik.networks import NN
from hessQuik.layers import resnetLayer
import hessQuik.activations as act
from copy import deepcopy


class resnetNN(NN):
    r"""
    Residual neural networks (ResNet) were popularized in the paper `Deep Residual Learning for Image Recognition`_ by He et al.
    Here, every layer is a single layer plus a skip connection.
    Let :math:`u_0` be the input into the ResNet.
    The construction is of the form

    .. _Deep Residual Learning for Image Recognition: https://ieeexplore.ieee.org/document/7780459

    .. math::

        \begin{align}
            u_1 &= u_0 + h\sigma(K_1 u_0 + b_1)\\
            u_2 &= u_1 + h\sigma(K_2 x + b_2)\\
                &\vdots \\
            u_{\ell} &= u_{\ell-1} + h\sigma(K_{\ell} u_{\ell-1} + b_{\ell})
        \end{align}

    where :math:`\ell` is the number of layers, called the depth of the network.
    Each vector of features :math:`u_i` is of size :math:`(n_s, w)` where :math:`n_s` is the number of samples
    and :math:`w` is the width of the network.
    Users choose the width and depth of the network and the activation function :math:`\sigma`.
    """

    def __init__(self, width: int, depth: int, h: float = 1.0,
                 act: act.hessQuikActivationFunction = act.softplusActivation(),
                 device=None, dtype=None):
        r"""

        :param width: dimension of hidden features
        :type width: int
        :param depth: number of ResNet layers
        :type depth: int
        :param h: step size, :math:`h > 0`. Default: 1.0
        :type h: float
        :param act: hessQuik activation function. Default: hessQuik.activations.softplusActivationFunction
        :type act: hessQuik.activations.hessQuikActivationFunction
        """
        factory_kwargs = {'device': device, 'dtype': dtype}

        args = ()
        for i in range(depth):
            args += (resnetLayer(width, h=h, act=deepcopy(act), **factory_kwargs),)

        super(resnetNN, self).__init__(*args)


if __name__ == '__main__':
    import torch
    from hessQuik.utils import input_derivative_check
    torch.set_default_dtype(torch.float64)

    # problem setup
    nex = 11
    d = 3
    x = torch.randn(nex, d)
    f = resnetNN(d, 4, h=0.5, act=act.softplusActivation())

    print('======= FORWARD =======')
    input_derivative_check(f, x, do_Hessian=True, verbose=True, forward_mode=True)

    print('======= BACKWARD =======')
    input_derivative_check(f, x, do_Hessian=True, verbose=True, forward_mode=False)
