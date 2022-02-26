import torch.nn as nn
from torch import Tensor
from typing import Union, Tuple


class hessQuikActivationFunction(nn.Module):
    """
    Forward propagate through a pointwise activation function, sigma(x)

    Parameters
    ----------
    x : (N, in_features) torch.Tensor
        Tensor of input features
    do_gradient : bool, optional
        If True, compute the gradient of the output of the layer, sigma(x), with respect to the layer input, x
    do_Hessian : bool, optional
        If True, compute the Hessian of the output of the layer, sigma(x), with respect to the layer input, x

    Returns
    -------
    sigma : (N, in_features) torch.Tensor
        Output of the layer, sigma(x)
    dsigma : (N, in_features)
        Gradient of the output of the layer, sigma(x), with respect to the layer inputs, x
    d2sigma : (N, in_features)
        Hessian of the output of the layer, sigma(x), with respect to the layer inputs, x
    """

    def __init__(self) -> None:
        super(hessQuikActivationFunction, self).__init__()
        self.ctx = None  # context variable

    def forward(self, x: Tensor, do_gradient: bool = False, do_Hessian: bool = False, forward_mode: bool = True) -> Tuple[Tensor, Union[Tensor, None], Union[Tensor, None]]:
        raise NotImplementedError

    def backward(self, do_Hessian: bool = False) -> Tuple[Tensor, Union[Tensor, None]]:
        """
        This is the backward method
        """
        dsigma, d2sigma = self.compute_derivatives(*self.ctx, do_Hessian=do_Hessian)

        return dsigma, d2sigma

    def compute_derivatives(self, *args, do_Hessian: bool = False) -> Tuple[Tensor, Union[Tensor, None]]:
        """
        This is the compute derivatives method
        """
        raise NotImplementedError
