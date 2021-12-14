import torch.nn as nn
from torch import Tensor
from typing import Union, Tuple


class hessQuikActivationFunction(nn.Module):

    def __init__(self) -> None:
        super(hessQuikActivationFunction, self).__init__()
        self.ctx = None  # context variable

    def forward(self, x: Tensor, do_gradient: bool = False, do_Hessian: bool = False, forward_mode: bool = True) -> \
            Tuple[Tensor, Union[Tensor, None], Union[Tensor, None]]:
        raise NotImplementedError

    def backward(self, do_Hessian: bool = False) -> Tuple[Tensor, Union[Tensor, None]]:
        dsigma, d2sigma = self.compute_derivatives(*self.ctx, do_Hessian=do_Hessian)

        return dsigma, d2sigma

    def compute_derivatives(self, *args, do_Hessian: bool = False) -> Tuple[Tensor, Union[Tensor, None]]:
        raise NotImplementedError
