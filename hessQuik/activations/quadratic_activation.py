import torch
from activation_function import activationFunction

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