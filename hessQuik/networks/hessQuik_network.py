import torch
import torch.nn as nn
from torch.autograd import grad
from torch.autograd.functional import hessian
from typing import Union, Tuple


class NN(nn.Sequential):
    r"""
    Wrapper for hessQuik networks built upon torch.nn.Sequential.
    """
    def __init__(self, *args):
        r"""
        :param args: sequence of hessQuik layers to be concatenated
        """
        # check for compatible composition
        for i, _ in enumerate(args[1:], start=1):
            n_out = args[i - 1].dim_output()
            n_in = args[i].dim_input()

            if not (n_out == n_in):
                raise ValueError("incompatible composition for block " + str(i - 1) + " to block " + str(i))

        super(NN, self).__init__(*args)

    def dim_input(self) -> int:
        r"""
        Number of network input features
        """
        return self[0].dim_input()

    def dim_output(self):
        r"""
        Number of network output features
        """
        return self[-1].dim_output()

    def setup_forward_mode(self, **kwargs):
        r"""
        Setup forward or backward mode.

        If ``kwargs`` does not include a ``forward_mode`` key, then the heuristic is to use ``forward_mode = True``
        if :math:`n_{in} < n_{out}` where :math:`n_{in}` is the number of input features and
        :math:`n_{out}` is the number of output features.

        There are three possible options once ``forward_mode`` is a key of ``kwargs``:

            - If ``forward_mode = True``, then the network computes derivatives during forward propagation.
            - If ``forward_mode = False``, then the network calls the backward routine to compute derivatives after forward propagating.
            - If ``forward_mode = None``, then the network will compute derivatives in backward mode, but will not call the backward routine.  This enables concatenation of networks, not just layers.
        """
        if not ('forward_mode' in kwargs.keys()):
            if self.dim_input() < self.dim_output():
                forward_mode = True  # compute the derivatives in forward mode
            else:
                forward_mode = False  # store necessary info, but do not compute derivatives until backward call
            kwargs['forward_mode'] = forward_mode

        return kwargs['forward_mode']

    def forward(self, x: torch.Tensor, do_gradient: bool = False, do_Hessian: bool = False,
                dudx: Union[torch.Tensor, None] = None, d2ud2x: Union[torch.Tensor, None] = None, **kwargs) \
            -> Tuple[torch.Tensor, Union[torch.Tensor, None], Union[torch.Tensor, None]]:
        r"""
        Forward propagate through network and compute derivatives

        :param x: input into network of shape :math:`(n_s, d)` where :math:`n_s` is the number of samples and :math:`d` is the number of input features
        :type x: torch.Tensor
        :param do_gradient: If set to ``True``, the gradient will be computed during the forward call. Default: ``False``
        :type do_gradient: bool, optional
        :param do_Hessian: If set to ``True``, the Hessian will be computed during the forward call. Default: ``False``
        :type do_Hessian: bool, optional
        :param dudx: if ``forward_mode = True``, gradient of features from previous layer with respect to network input :math:`x` with shape :math:`(n_s, d, n_{in})`
        :type dudx: torch.Tensor or ``None``
        :param d2ud2x: if ``forward_mode = True``, Hessian of features from previous layer with respect to network input :math:`x` with shape :math:`(n_s, d, d, n_{in})`
        :type d2ud2x: torch.Tensor or ``None``
        :param kwargs: additional options, such as ``forward_mode`` as a user input
        :return:

            - **f** (*torch.Tensor*) - output features of network with shape :math:`(n_s, m)` where :math:`m` is the number of network output features
            - **dfdx** (*torch.Tensor* or ``None``) - if ``forward_mode = True``, gradient of output features with respect to network input :math:`x` with shape :math:`(n_s, d, m)`
            - **d2fd2x** (*torch.Tensor* or ``None``) - if ``forward_mode = True``, Hessian of output features with respect to network input :math:`x` with shape :math:`(n_s, d, d, m)`
        """
        forward_mode = self.setup_forward_mode(**kwargs)

        for module in self:
            x, dudx, d2ud2x = module(x, do_gradient=do_gradient, do_Hessian=do_Hessian, dudx=dudx, d2ud2x=d2ud2x,
                                     forward_mode=True if forward_mode is True else None)

        if (do_gradient or do_Hessian) and forward_mode is False:
            dudx, d2ud2x = self.backward(do_Hessian=do_Hessian)

        return x, dudx, d2ud2x

    def backward(self, do_Hessian: bool = False, dgdf: Union[torch.Tensor, None] = None,
                 d2gd2f: Union[torch.Tensor, None] = None) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        r"""
        Compute derivatives using backward propagation.  This method is called during the forward pass if ``forward_mode = False``.

        :param do_Hessian: If set to ``True``, the Hessian will be computed during the forward call. Default: ``False``
        :type do_Hessian: bool, optional
        :param dgdf: gradient of the subsequent layer features, :math:`g(f)`, with respect to the layer outputs, :math:`f` with shape :math:`(n_s, n_{out}, m)`.
        :type dgdf: torch.Tensor
        :param d2gd2f: gradient of the subsequent layer features, :math:`g(f)`, with respect to the layer outputs, :math:`f` with shape :math:`(n_s, n_{out}, n_{out}, m)`.
        :type d2gd2f: torch.Tensor or ``None``
        :return:

            - **dgdf** (*torch.Tensor* or ``None``) - gradient of the network with respect to input features :math:`x` with shape :math:`(n_s, d, m)`
            - **d2gd2f** (*torch.Tensor* or ``None``) - Hessian of the network with respect to input features :math:`u` with shape :math:`(n_s, d, d, m)`

        """
        for i in range(len(self) - 1, -1, -1):
            dgdf, d2gd2f = self[i].backward(do_Hessian=do_Hessian, dgdf=dgdf, d2gd2f=d2gd2f)
        return dgdf, d2gd2f


class NNPytorchAD(nn.Module):
    r"""
    Compute the derivatives of a network using Pytorch's automatic differentiation.

    The implementation follows that of `CP Flow`_.

    .. _CP Flow: https://github.com/CW-Huang/CP-Flow
    """

    def __init__(self, net: NN):
        r"""
        Create wrapper around hessQuik network.

        :param net: hessQuik network
        :type net: hessQuik.networks.NN
        """
        super(NNPytorchAD, self).__init__()
        self.net = net

    def forward(self, x: torch.Tensor, do_gradient: bool = False, do_Hessian: bool = False, **kwargs) \
            -> Tuple[torch.Tensor, Union[torch.Tensor, None], Union[torch.Tensor, None]]:
        r"""
        Forward propagate through the hessQuik network without computing derivatives.
        Then, use automatic differentiation to compute derivatives using ``torch.autograd.grad``.
        """

        (df, d2f) = (None, None)
        if do_gradient or do_Hessian:
            x.requires_grad = True

        # forwaard propagate without compute derivatives
        f, *_ = self.net(x, do_gradient=False, do_Hessian=False, forward_mode=False)

        if do_gradient or do_Hessian:
            f = f.view(x.shape[0], -1)
            df = []
            for j in range(f.shape[1]):
                df.append(grad(f[:, j].sum(), x, create_graph=True, retain_graph=True)[0])
            df = torch.stack(df, dim=2)

            if do_Hessian:
                df = df.reshape(x.shape[0], -1)
                d2f = []
                for j in range(df.shape[1]):
                    d2f.append(grad(df[:, j].sum(), x, create_graph=True, retain_graph=True)[0])
                d2f = torch.stack(d2f, dim=2)
                d2f = d2f.reshape(x.shape[0], x.shape[1], x.shape[1], -1).squeeze(-1)
                if d2f.dim() < 4:
                    d2f = d2f.unsqueeze(-1)

            df = df.reshape(x.shape[0], x.shape[1], -1).squeeze(-1)

            if df.dim() < 3:
                df = df.unsqueeze(-1)

        return f, df, d2f


class NNPytorchHessian(nn.Module):
    r"""
    Compute the derivatives of a network using Pytorch's Hessian functional.
    """

    def __init__(self, net):
        """
        Create wrapper around hessQuik network.

        :param net: hessQuik network
        :type net: hessQuik.networks.NN
        """
        super(NNPytorchHessian, self).__init__()
        self.net = net

    def forward(self, x: torch.Tensor, do_gradient: bool = False, do_Hessian: bool = False, **kwargs) \
            -> Tuple[torch.Tensor, Union[torch.Tensor, None], Union[torch.Tensor, None]]:
        r"""
        Forward propagate through the hessQuik network without computing derivatives.
        Then, use automatic differentiation to compute derivatives using ``torch.autograd.functional.hessian``.
        """

        (df, d2f) = (None, None)
        if do_gradient or do_Hessian:
            x.requires_grad = True

        f, *_ = self.net(x, do_gradient=False, do_Hessian=False, forward_mode=False)

        if f.squeeze().ndim > 1:
            raise ValueError(type(self), " must have scalar outputs per example")

        if do_gradient:
            df = grad(f.sum(), x)[0]
            df = df.unsqueeze(-1)

        if do_Hessian:
            d2f = hessian(lambda x: self.net(x)[0].sum(), x).sum(dim=2)
            d2f = d2f.unsqueeze(-1)

        return f, df, d2f


if __name__ == '__main__':
    import torch
    import hessQuik.activations as act
    import hessQuik.layers as lay
    from hessQuik.utils import input_derivative_check
    torch.set_default_dtype(torch.float64)

    # problem setup
    nex = 11
    d = 3
    ms = [2, 7, 5]
    m = 8
    x = torch.randn(nex, d)

    f = NN(lay.singleLayer(d, ms[0], act=act.softplusActivation()),
           lay.singleLayer(ms[0], ms[1], act=act.softplusActivation()),
           lay.singleLayer(ms[1], ms[2], act=act.softplusActivation()),
           lay.singleLayer(ms[2], m, act=act.softplusActivation()))

    # f = NNPytorchHessian(f)
    # x.requires_grad = True

    print('======= FORWARD =======')
    input_derivative_check(f, x, do_Hessian=True, verbose=True, forward_mode=True)

    print('======= BACKWARD =======')
    input_derivative_check(f, x, do_Hessian=True, verbose=True, forward_mode=False)
