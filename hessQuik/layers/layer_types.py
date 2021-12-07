import torch
import torch.nn as nn
from torch import Tensor
from typing import Union, Tuple


class activationFunction(torch.nn.Module):

    def __init__(self) -> None:
        super(activationFunction, self).__init__()
        self.ctx = None  # context variable

    def forward(self, x: Tensor, do_gradient: bool = False, do_Hessian: bool = False, reverse_mode: bool = False) -> \
            Tuple[Tensor, Union[Tensor, None], Union[Tensor, None]]:
        raise NotImplementedError

    def backward(self, do_Hessian: bool = False) -> Tuple[Tensor, Union[Tensor, None]]:
        raise NotImplementedError


class hessQuikLayer(nn.Module):

    def __init__(self, *args, **kwargs):
        super(hessQuikLayer, self).__init__()

    def forward(self, u: torch.Tensor, do_gradient: bool = False, do_Hessian: bool = False,
                dudx: Union[torch.Tensor, None] = None, d2ud2x: Union[torch.Tensor, None] = None,
                reverse_mode: bool = False) \
            -> Tuple[torch.Tensor, Union[torch.Tensor, None], Union[torch.Tensor, None]]:
        raise NotImplementedError

    def backward(self, do_Hessian: bool = False,
                 dgdf: Union[torch.Tensor, None] = None, d2gd2f: Union[torch.Tensor, None] = None) \
            -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        raise NotImplementedError
