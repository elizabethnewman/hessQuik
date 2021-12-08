import torch
from activation_function import activationFunction


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