import torch
from hessQuik.activations import activationFunction

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


if __name__ == '__main__':
    from hessQuik.tests import DerivativeCheckTestsActivationFunction
    torch.set_default_dtype(torch.float64)

    nex = 11  # no. of examples
    d = 4  # no. of input features

    x = torch.randn(nex, d)
    dx = torch.randn_like(x)

    f = quadraticActivation()

    derivativeTests = DerivativeCheckTestsActivationFunction()

    print('======= FORWARD =======')
    derivativeTests.run_forward_hessian_test(f, x, dx, verbose=True)

    print('======= BACKWARD =======')
    derivativeTests.run_backward_hessian_test(f, x, dx, verbose=True)
