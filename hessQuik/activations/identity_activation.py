import torch
from activation_function import activationFunction

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
