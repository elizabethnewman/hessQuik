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

<<<<<<< HEAD
<<<<<<< HEAD
    def forward(self, u: Tensor, do_gradient: bool = False, do_Hessian: bool = False, do_Laplacian: bool = False,
                dudx: Union[Tensor, None] = None, d2ud2x: Union[Tensor, None] = None,
                lap_u: Union[Tensor, None] = None) \
            -> Tuple[Tensor, Union[Tensor, None], Union[Tensor, None], Union[Tensor, None]]:
        """
        Forward propagate through singleLayer

        Parameters
        ----------
        u : (N, in_features) torch.Tensor
            Tensor of input features
        do_gradient : bool, optional
            If True, compute the gradient of the output of the layer, f(u(x)), with respect to the network input, x
        do_Hessian : bool, optional
            If True, compute the Hessian of the output of the layer, f(u(x)), with respect to the network input, x
        dudx : (N, d, in_features) torch.Tensor
            Gradient of the input into the layer, u(x), with respect to the network inputs, x
        d2ud2x : (N, d, d, in_features) torch.Tensor
            Hessian of the input into the layer, u(x), with respect to the network inputs, x

        Returns
        -------
        f : (N, out_features) torch.Tensor
            Output of the layer, f(u(x))
        dfdx : (N, d, out_features)
            Gradient of the output of the layer, f(u(x)), with respect to the network inputs, x
        d2fd2x : (N, d, d, out_features)
            Hessian of the output of the layer, f(u(x)), with respect to the network inputs, x
        lap_f  : (N, out_features)
            Laplacian of the output of the layer, f(u(x)), with respect to the network inputs, x
        """
        raise NotImplementedError

    def backward(self, do_Hessian: bool = False, do_Laplcian: bool = False, dgdf: Union[Tensor, None] = None,
                 d2gd2f: Union[Tensor, None] = None, lap_g: Union[Tensor, None] = None) \
            -> Tuple[Tensor, Union[Tensor, None], Union[Tensor, None]]:
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
        lap_g : (N, in_features)
            Laplacian of the output of the layer, g(f(x)), with respect to the network inputs, x
        """
=======
    def forward(self, u: Tensor, do_gradient: bool = False, do_Hessian: bool = False, forward_mode: bool = True,
                dudx: Union[Tensor, None] = None, d2ud2x: Union[Tensor, None] = None) \
=======
    def forward(self, u: Tensor,
                do_gradient: bool = False, do_Hessian: bool = False, do_Laplacian: bool = False,
                forward_mode: bool = True,
                dudx: Union[Tensor, None] = None, d2ud2x: Union[Tensor, None] = None,
                v: Union[Tensor, None] = None) \
>>>>>>> main
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
        :param do_Hessian: If set to ``True``, the Hessian will be computed during the forward call. Default: ``False``
        :type do_Hessian: bool, optional
        :param do_Laplacian: If set to ``True``, the Laplacian will be computed during the forward call. Default: ``False``
        :type do_Laplacian: bool, optional
        :param forward_mode:  If set to ``False``, the derivatives will be computed in backward mode. Default: ``True``
        :type forward_mode: bool, optional
        :param dudx: if ``forward_mode = True``, gradient of features from previous layer with respect to network input :math:`x` with shape :math:`(n_s, d, n_{in})`
        :type dudx: torch.Tensor or ``None``
        :param d2ud2x: if ``forward_mode = True``, Hessian of features from previous layer with respect to network input :math:`x` with shape :math:`(n_s, d, d, n_{in})`
        :type d2ud2x: torch.Tensor or ``None``
        :param v: if ``forward_mode = True``, direction(s) to apply Jacobian and Hessian :math:`x` with shape :math:`(d, k)`
        :type v: torch.Tensor or ``None``
        :return:

            - **f** (*torch.Tensor*) - output features of layer with shape :math:`(n_s, n_{out})`
            - **dfdx** (*torch.Tensor* or ``None``) - if ``forward_mode = True``, gradient of output features with respect to network input :math:`x` with shape :math:`(n_s, d, n_{out})`
            - **d2fd2x** (*torch.Tensor* or ``None``) - if ``forward_mode = True``, Hessian of output features with respect to network input :math:`x` with shape :math:`(n_s, d, d, n_{out})`

        """

        raise NotImplementedError

    def backward(self, do_Hessian: bool = False,
                 dgdf: Union[Tensor, None] = None, d2gd2f: Union[Tensor, None] = None,
                 v: Union[Tensor, None] = None) \
            -> Tuple[Tensor, Union[Tensor, None]]:
        r"""
        Backward pass through the layer that maps the gradient of the network :math:`g`
        with respect to the output features :math:`f` of size :math:`(n_s, n_{out}, m)`
        to the gradient of the network :math:`g` with respect to the input features :math:`u` of size :math:`(n_s, n_{in}, m)`
        where :math:`n_s` is the number of samples, :math:`n_{in}` is the number of input features,,
        :math:`n_{out}` is the number of output features, and :math:`m` is the number of network output features.

        :param do_Hessian: If set to ``True``, the Hessian will be computed during the forward call. Default: ``False``
        :type do_Hessian: bool, optional
        :param dgdf: gradient of the subsequent layer features, :math:`g(f)`, with respect to the layer outputs, :math:`f` with shape :math:`(n_s, n_{out}, m)`.
        :type dgdf: torch.Tensor
        :param d2gd2f: gradient of the subsequent layer features, :math:`g(f)`, with respect to the layer outputs, :math:`f` with shape :math:`(n_s, n_{out}, n_{out}, m)`.
        :type d2gd2f: torch.Tensor or ``None``
        :param v: direction(s) to apply Jacobian transpose and Hessian :math:`x` with shape :math:`(d, k)`
        :type v: torch.Tensor or ``None``
        :return:

            - **dgdu** (*torch.Tensor* or ``None``) -  gradient of the network with respect to input features :math:`u` with shape :math:`(n_s, n_{in}, m)`
            - **d2gd2u** (*torch.Tensor* or ``None``) - Hessian of the network with respect to input features :math:`u` with shape :math:`(n_s, n_{in}, n_{in}, m)`

        """

>>>>>>> c846faf2d50607569f3f073aa019d49e967371c4
        raise NotImplementedError


