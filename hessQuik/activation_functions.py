import torch
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


class antiTanhActivation(activationFunction):

    def __init__(self):
        super(antiTanhActivation, self).__init__()

    def forward(self, x, do_gradient=False, do_Hessian=False, reverse_mode=False):
        (dsigma, d2sigma) = (None, None)

        sigma = torch.abs(x) + torch.log(1 + torch.exp(-2.0 * torch.abs(x)))

        if reverse_mode:
            self.ctx = (x,)
        else:
            if do_gradient or do_Hessian:
                dsigma = torch.tanh(x)
                if do_Hessian:
                    d2sigma = 1 - dsigma ** 2

        return sigma, dsigma, d2sigma

    def backward(self, do_Hessian=False):
        x, = self.ctx
        d2sigma = None
        dsigma = torch.tanh(x)
        if do_Hessian:
            d2sigma = 1 - dsigma ** 2
        return dsigma, d2sigma


class quadraticActivation(activationFunction):

    def __init__(self):
        super(quadraticActivation, self).__init__()

    def forward(self, x, do_gradient=False, do_Hessian=False, reverse_mode=False):
        (dsigma, d2sigma) = (None, None)

        if reverse_mode:
            self.ctx = (x,)

        sigma = 0.5 * (x ** 2)

        if do_gradient:
            dsigma = x

        if do_Hessian:
            d2sigma = torch.ones_like(x)

        return sigma, dsigma, d2sigma

    def backward(self, do_Hessian=False):
        x, = self.ctx
        d2sigma = None
        if do_Hessian:
            d2sigma = torch.ones_like(x)
        return x, d2sigma


class softplusActivation(activationFunction):

    def __init__(self, beta=1, threshold=20):
        super(softplusActivation, self).__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, x, do_gradient=False, do_Hessian=False, reverse_mode=False):
        (dsigma, d2sigma) = (None, None)
        # implement ourselves!
        # sigma = F.softplus(x)
        # dsigma = torch.exp(x) / (1 + torch.exp(x))
        # d2sigma = torch.exp(x) / ((1 + torch.exp(x))**2)
        if reverse_mode:
            self.ctx = (x,)

        # initialize sigma
        sigma = torch.clone(x)

        # find values of x below threshold
        idx = self.beta * x < self.threshold
        sigma[idx] = (1 / self.beta) * torch.log(1 + torch.exp(self.beta * x[idx]))

        if do_gradient or do_Hessian:
            idx = self.beta * x < self.threshold
            dsigma = torch.ones_like(x)
            dsigma[idx] = torch.exp(self.beta * x[idx]) / (1 + torch.exp(self.beta * x[idx]))

            if do_Hessian:
                d2sigma = torch.zeros_like(x)
                d2sigma[idx] = self.beta * torch.exp(self.beta * x[idx]) / ((1 + torch.exp(self.beta * x[idx])) ** 2)

        return sigma, dsigma, d2sigma

    def backward(self, do_Hessian=False):
        x, = self.ctx
        d2sigma = None

        idx = self.beta * x < self.threshold
        dsigma = torch.ones_like(x)
        dsigma[idx] = torch.exp(self.beta * x[idx]) / (1 + torch.exp(self.beta * x[idx]))

        if do_Hessian:
            d2sigma = torch.zeros_like(x)
            d2sigma[idx] = self.beta * torch.exp(self.beta * x[idx]) / ((1 + torch.exp(self.beta * x[idx])) ** 2)

        return dsigma, d2sigma


class identityActivation(activationFunction):

    def __init__(self):
        super(identityActivation, self).__init__()

    def forward(self, x, do_gradient=False, do_Hessian=False, reverse_mode=False):
        (dsigma, d2sigma) = (None, None)

        if reverse_mode:
            self.ctx = (x,)

        sigma = x

        if do_gradient:
            dsigma = torch.ones_like(x)

        if do_Hessian:
            d2sigma = torch.zeros_like(x)

        return sigma, dsigma, d2sigma

    def backward(self, do_Hessian=False):
        x, = self.ctx
        d2sigma = None
        if do_Hessian:
            d2sigma = torch.zeros_like(x)
        return torch.ones_like(x), d2sigma
