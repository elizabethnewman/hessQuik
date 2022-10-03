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

<<<<<<< HEAD
    @property
    def reverse_mode(self):
        return self._reverse_mode

    @reverse_mode.setter
    def reverse_mode(self, reverse_mode):
        self._reverse_mode = reverse_mode

    def forward(self, x: Tensor, do_gradient: bool = False, do_Hessian: bool = False, do_Laplacian: bool = False) -> \
            Tuple[Tensor, Union[Tensor, None], Union[Tensor, None]]:
        raise NotImplementedError

    def backward(self, do_Hessian: bool = False, do_Laplacian: bool = False) -> Tuple[Tensor, Union[Tensor, None]]:
        dsigma, d2sigma = self.compute_derivatives(*self.ctx, do_Hessian=do_Hessian, do_Laplacian=do_Laplacian)

        return dsigma, d2sigma

    def compute_derivatives(self, *args, do_Hessian: bool = False,  do_Laplacian: bool = False) \
            -> Tuple[Tensor, Union[Tensor, None]]:
=======
    def forward(self, x: Tensor, do_gradient: bool = False, do_Hessian: bool = False, forward_mode: bool = True) \
            -> Tuple[Tensor, Union[Tensor, None], Union[Tensor, None]]:
        r"""
        Applies a pointwise activation function to the incoming data.

        :param x: input into the activation function. :math:`(*)` where :math:`*` means any shape.
        :type x: torch.Tensor
        :param do_gradient: If set to ``True``, the gradient will be computed during the forward call. Default: ``False``
        :type do_gradient: bool, optional
        :param do_Hessian: If set to ``True``, the Hessian will be computed during the forward call. Default: ``False``
        :type do_Hessian: bool, optional
        :param forward_mode:  If set to ``False``, the derivatives will be computed in backward mode. Default: ``True``
        :type forward_mode: bool, optional
        :return:

            - **sigma** (*torch.Tensor*) - value of activation function at input x, same size as x
            - **dsigma** (*torch.Tensor* or ``None``) - first derivative of activation function at input x, same size as x
            - **d2sigma** (*torch.Tensor* or ``None``) - second derivative of activation function at input x, same size as x

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

        :param args: variables needed to compute derivatives
        :type args: torch.Tensor
        :param do_Hessian: If set to ``True``, the Hessian will be computed during the forward call. Default: ``False``
        :type do_Hessian: bool, optional
        :return:

            - **dsigma** (*torch.Tensor* or ``None``) - first derivative of activation function at input x, same size as x
            - **d2sigma** (*torch.Tensor* or ``None``) - second derivative of activation function at input x, same size as x

        """
>>>>>>> c846faf2d50607569f3f073aa019d49e967371c4
        raise NotImplementedError
