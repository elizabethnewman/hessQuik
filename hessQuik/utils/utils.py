import torch
from torch import Tensor
from torch.nn import Module
from typing import Tuple, List
from math import log2, floor


def module_getattr(obj: Module, names: Tuple or List or str):
    if isinstance(names, str) or len(names) == 1:
        if len(names) == 1:
            names = names[0]

        return getattr(obj, names)
    else:
        return module_getattr(getattr(obj, names[0]), names[1:])


def module_setattr(obj: Module, names: Tuple or List, val: Tensor):
    if isinstance(names, str) or len(names) == 1:
        if len(names) == 1:
            names = names[0]

        return setattr(obj, names, val)
    else:
        return module_setattr(getattr(obj, names[0]), names[1:], val)


def extract_data(net: Module, attr: str = 'data') -> (Tensor, Tuple, Tuple):
    theta = torch.empty(0)
    for name, w in net.named_parameters():
        if getattr(w, attr) is None:
            w = torch.zeros_like(w.data)
        else:
            w = getattr(w, attr)

        theta = torch.cat((theta, w.reshape(-1)))

    return theta


def insert_data(net: Module, theta: Tensor) -> None:

    count = 0
    for name, w in net.named_parameters():
        name_split = name.split('.')
        n = w.numel()
        module_setattr(net, name_split + ['data'], theta[count:count + n].reshape(w.shape))
        count += n


def convert_to_base(a, b=2.0):
    outputs = ()
    for i in range(len(a)):
        if a[i] <= 0:
            # catch case when equal to 0
            c, d = -1, 0
        else:
            d = floor(log2(a[i]) / log2(b))
            c = b ** (log2(a[i]) / log2(b) - d)

        outputs += (c, d)

    return outputs