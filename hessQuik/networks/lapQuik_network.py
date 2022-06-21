import torch
import torch.nn as nn
from typing import Union, Tuple


class NNLapQuik(nn.Sequential):
    r"""
    Wrapper for lapQuik networks built upon torch.nn.Sequential.
    """
    def __init__(self, *args):
        r"""
        :param args: sequence of lapQuik layers to be concatenated
        """
        # check for compatible composition
        for i, _ in enumerate(args[1:], start=1):
            n_out = args[i - 1].dim_output()
            n_in = args[i].dim_input()

            if not (n_out == n_in):
                raise ValueError("incompatible composition for block " + str(i - 1) + " to block " + str(i))

        super(NNLapQuik, self).__init__(*args)

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

    def forward(self, x: torch.Tensor, do_gradient: bool = False, do_Laplacian: bool = False,
                dudx: Union[torch.Tensor, None] = None, lapud2x: Union[torch.Tensor, None] = None, **kwargs) \
            -> Tuple[torch.Tensor, Union[torch.Tensor, None], Union[torch.Tensor, None]]:
        r"""
        Forward propagate through network and compute derivatives

        :param x: input into network of shape :math:`(n_s, d)` where :math:`n_s` is the number of samples and :math:`d` is the number of input features
        :type x: torch.Tensor
        :param do_gradient: If set to ``True``, the gradient will be computed during the forward call. Default: ``False``
        :type do_gradient: bool, optional
        :param do_Laplacian: If set to ``True``, the Hessian will be computed during the forward call. Default: ``False``
        :type do_Laplacian: bool, optional
        :param dudx: if ``forward_mode = True``, gradient of features from previous layer with respect to network input :math:`x` with shape :math:`(n_s, d, n_{in})`
        :type dudx: torch.Tensor or ``None``
        :param lapud2x: if ``forward_mode = True``, Hessian of features from previous layer with respect to network input :math:`x` with shape :math:`(n_s, d, d, n_{in})`
        :type lapud2x: torch.Tensor or ``None``
        :param kwargs: additional options, such as ``forward_mode`` as a user input
        :return:

            - **f** (*torch.Tensor*) - output features of network with shape :math:`(n_s, m)` where :math:`m` is the number of network output features
            - **dfdx** (*torch.Tensor* or ``None``) - if ``forward_mode = True``, gradient of output features with respect to network input :math:`x` with shape :math:`(n_s, d, m)`
            - **d2fd2x** (*torch.Tensor* or ``None``) - if ``forward_mode = True``, Hessian of output features with respect to network input :math:`x` with shape :math:`(n_s, d, d, m)`
        """

        for module in self:
            x, dudx, lapud2x = module(x, do_gradient=do_gradient, do_Laplacian=do_Laplacian, dudx=dudx, lapud2x=lapud2x)

        return x, dudx, lapud2x


if __name__ == '__main__':
    import torch
    import hessQuik.activations as act
    import hessQuik.layers as lay
    from hessQuik.utils import input_derivative_check_finite_difference_laplacian
    torch.set_default_dtype(torch.float64)

    # problem setup
    nex = 11
    d = 3
    ms = [2, 7, 5]
    m = 8
    x = torch.randn(nex, d)

    f = NNLapQuik(lay.singleLayerLapQuik(d, ms[0], act=act.softplusActivation()),
                  lay.singleLayerLapQuik(ms[0], ms[1], act=act.softplusActivation()),
                  lay.singleLayerLapQuik(ms[1], ms[2], act=act.softplusActivation()),
                  lay.singleLayerLapQuik(ms[2], m, act=act.softplusActivation()))

    x.requires_grad = True

    input_derivative_check_finite_difference_laplacian(f, x, do_Laplacian=True, verbose=True)

