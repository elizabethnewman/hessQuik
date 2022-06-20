import torch.nn as nn
from torch import Tensor
from typing import Union, Tuple


class lapQuikLayer(nn.Module):
    r"""
    Base class for all hessQuik layers.
    """

    def __init__(self, *args, **kwargs):
        super(lapQuikLayer, self).__init__()

    def dim_input(self) -> int:
        r"""

        :return: dimension of input features
        :rtype: int
        """
        raise NotImplementedError

    def dim_output(self) -> int:
        r"""

        :return: dimension of output features
        :rtype: int
        """
        raise NotImplementedError

    def forward(self, u: Tensor, do_gradient: bool = False, do_Laplacian: bool = False,
                dudx: Union[Tensor, None] = None, lapud2x: Union[Tensor, None] = None) \
            -> Tuple[Tensor, Union[Tensor, None], Union[Tensor, None]]:
        r"""
        Forward pass through the layer that maps input features :math:`u` of size :math:`(n_s, n_{in})`
        to output features :math:`f` of size :math:`(n_s, n_{out})` where :math:`n_s` is the number of samples,
        :math:`n_{in}` is the number of input features, and :math:`n_{out}` is the number of output features.

        The input features :math:`u(x)` is a function of the network input :math:`x` of size :math:`(n_s, d)`
        where :math:`d` is the dimension of the network input.

        :param u: features from previous layer with shape :math:`(n_s, n_{in})`
        :type u: torch.Tensor
        :param do_gradient: If set to ``True``, the gradient will be computed during the forward call. Default: ``False``
        :type do_gradient: bool, optional
        :param do_Laplacian: If set to ``True``, the Hessian will be computed during the forward call. Default: ``False``
        :type do_Laplacian: bool, optional
        :param dudx: gradient of features from previous layer with respect to network input :math:`x` with shape :math:`(n_s, d, n_{in})`
        :type dudx: torch.Tensor or ``None``
        :param lapud2x: Laplacian of features from previous layer with respect to network input :math:`x` with shape :math:`(n_s, 1, n_{in})`
        :type lapud2x: torch.Tensor or ``None``
        :return:

            - **f** (*torch.Tensor*) - output features of layer with shape :math:`(n_s, n_{out})`
            - **dfdx** (*torch.Tensor* or ``None``) - gradient of output features with respect to network input :math:`x` with shape :math:`(n_s, d, n_{out})`
            - **lapfd2x** (*torch.Tensor* or ``None``) - Laplacian of output features with respect to network input :math:`x` with shape :math:`(n_s, 1, n_{out})`

        """

        raise NotImplementedError



