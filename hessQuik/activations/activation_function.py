import torch.nn as nn
from torch import Tensor
from typing import Union, Tuple




class activationFunction(nn.Module):

    def __init__(self) -> None:
        super(activationFunction, self).__init__()
        self.ctx = None  # context variable

    def forward(self, x: Tensor, do_gradient: bool = False, do_Hessian: bool = False, reverse_mode: bool = False) -> \
            Tuple[Tensor, Union[Tensor, None], Union[Tensor, None]]:
        raise NotImplementedError

    def backward(self, do_Hessian: bool = False) -> Tuple[Tensor, Union[Tensor, None]]:
        raise NotImplementedError
