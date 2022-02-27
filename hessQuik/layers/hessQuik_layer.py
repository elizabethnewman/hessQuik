import torch.nn as nn
from torch import Tensor
from typing import Union, Tuple


class hessQuikLayer(nn.Module):
    r"""
    Base class for all hessQuik layers.
    """

    def __init__(self, *args, **kwargs):
        super(hessQuikLayer, self).__init__()

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

    def forward(self, u: Tensor, do_gradient: bool = False, do_Hessian: bool = False, forward_mode: bool = True,
                dudx: Union[Tensor, None] = None, d2ud2x: Union[Tensor, None] = None) \
            -> Tuple[Tensor, Union[Tensor, None], Union[Tensor, None]]:
        r"""
        Forward pass through the layer that maps
        .. math::

            \text{input features :math:`u` of size :math:`(n_s, n_{in})`} \longleftarrow
            output features :math:`f` of size :math:`(n_s, n_{out})`

        where :math:`n_s` is the number of samples and :math:`n_{in}` is the number of input features,
        and :math:`n_{out}` is the number of output features.

        The input features :math:`u(x)` is a function of the network input :math:`x` of size :math:`(n_s, d)`
        where :math:`d` is the dimension of the network input.

        :param u: features from previous layer. :math:`(n_s, n_{in})`
        :type u: torch.Tensor
        :param do_gradient: If set to ``True``, the gradient will be computed during the forward call. Default: ``False``
        :type do_gradient: bool, optional
        :param do_Hessian: If set to ``True``, the Hessian will be computed during the forward call. Default: ``False``
        :type do_Hessian: bool, optional
        :param forward_mode:  If set to ``False``, the derivatives will be computed in backward mode. Default: ``True``
        :type forward_mode: bool, optional
        :param dudx: gradient of features from previous layer with respect to network input x. :math:`(n_s, d, n_{in})`
        :type dudx: torch.Tensor or ``None``
        :param d2ud2x: Hessian of features from previous layer with respect to network input x. :math:`(n_s, d, d, n_{in})`
        :type d2ud2x: torch.Tensor or ``None``
        :return:

            - **f** (*torch.Tensor*) - output features of layer. :math:`(n_s, n_{out})`
            - **df** (*torch.Tensor* or ``None``) - if ``forward_mode = True``, gradient of output features with respect to network input x. :math:`(n_s, d, n_{out})`
            - **d2f** (*torch.Tensor* or ``None``) - if ``forward_mode = True``, Hessian of output features with respect to network input x. :math:`(n_s, d, d, n_{out})`

        """

        raise NotImplementedError

    def backward(self, do_Hessian: bool = False,
                 dgdf: Union[Tensor, None] = None, d2gd2f: Union[Tensor, None] = None) \
            -> Tuple[Tensor, Union[Tensor, None]]:
        """
        Backward propagate through singleLayer

        Parameters
        ----------
        do_Hessian : bool, optional
            If True, compute the Hessian of the output of the layer, g(f(x)), with respect to the network input, x
        dgdf : (N, d, out_features) torch.Tensor
            Gradient of the input into the layer, g(f(x)), with respect to the layer outputs, f(x)
        d2gd2f : (N, d, d, out_features) torch.Tensor
            Hessian of the input into the layer, g(f(x)), with respect to the layer outputs, f(x)

        Returns
        -------
        dgdx : (N, d, in_features)
            Gradient of the output of the layer, g(f(x)), with respect to the network inputs, x
        d2gd2x : (N, d, d, in_features)
            Hessian of the output of the layer, g(f(x)), with respect to the network inputs, x
        """
        raise NotImplementedError


