import torch.nn as nn
from torch import Tensor
from typing import Union, Tuple


class hessQuikLayer(nn.Module):

    def __init__(self, *args, **kwargs):
        super(hessQuikLayer, self).__init__()

    def forward(self, u: Tensor, do_gradient: bool = False, do_Hessian: bool = False,
                dudx: Union[Tensor, None] = None, d2ud2x: Union[Tensor, None] = None,
                reverse_mode: bool = False) \
            -> Tuple[Tensor, Union[Tensor, None], Union[Tensor, None]]:
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


