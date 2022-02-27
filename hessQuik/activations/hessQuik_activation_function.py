import torch.nn as nn
from torch import Tensor
from typing import Union, Tuple


class hessQuikActivationFunction(nn.Module):
    r"""
    Base class for all hessQuik activation functions.
    """

    def __init__(self) -> None:
        super(hessQuikActivationFunction, self).__init__()
        self.ctx = None  # context variable

    def forward(self, x: Tensor, do_gradient: bool = False, do_Hessian: bool = False, forward_mode: bool = True) \
            -> Tuple[Tensor, Union[Tensor, None], Union[Tensor, None]]:
        r"""
        Applies a pointwise activation function to the incoming data.

        Input:

            x (torch.Tensor): input into the activation function. :math:`(*)` where :math:`*` means any size.


            do_gradient (bool): If set to ``True``, the gradient will be computed during the forward call.
                                Default: ``True``

            do_Hessian (bool): If set to ``True``, the Hessian will be computed during the forward call.
                                Default: ``True``

            forward_mode (bool): If set to ``False``, the derivatives will be computed in backward mode.
                                Default: ``True``

        Return:

            sigma (torch.Tensor): value of activation function at input x, same size as x

            dsigma (torch.Tensor): optional, first derivative of activation function at input x, same size as x

            d2sigma (torch.Tensor): optional, second derivative of activation function at input x, same size as x
        """
        raise NotImplementedError

    def backward(self, do_Hessian: bool = False) -> Tuple[Tensor, Union[Tensor, None]]:
        r"""
        Computes derivatives of activation function evaluated at x in backward mode.

        Calls self.compute_derivatives without inputs, stores necessary variables in self.ctx.

        Inherited by all subclasses.
        """
        dsigma, d2sigma = self.compute_derivatives(*self.ctx, do_Hessian=do_Hessian)

        return dsigma, d2sigma

    def compute_derivatives(self, *args, do_Hessian: bool = False) -> Tuple[Tensor, Union[Tensor, None]]:
        r"""
        Computes derivatives of activation function evaluated at x in either forward or backward more

        Input:

            *args (tuple): tuple of variables needed to compute derivatives

            do_Hessian (bool): If set to ``True``, the Hessian will be computed during the forward call.
                                Default: ``True``

        Return:

            dsigma (torch.Tensor): optional, first derivative of activation function at input x, same size as x

            d2sigma (torch.Tensor): optional, second derivative of activation function at input x, same size as x
        """
        raise NotImplementedError
