import torch
from hessQuik.activations import activationFunction


class tanhActivation(activationFunction):

    def __init__(self):
        super(tanhActivation, self).__init__()

    def forward(self, x, do_gradient=False, do_Hessian=False, reverse_mode=False):
        (dsigma, d2sigma) = (None, None)

        sigma = torch.tanh(x)

        if reverse_mode:
            self.ctx = (sigma,)
        else:
            if do_gradient or do_Hessian:
                dsigma = 1 - sigma ** 2
                if do_Hessian:
                    d2sigma = -2 * sigma * (1 - sigma ** 2)

        return sigma, dsigma, d2sigma

    def backward(self, do_Hessian=False):
        sigma, = self.ctx
        d2sigma = None
        dsigma = 1 - sigma ** 2
        if do_Hessian:
            d2sigma = -2 * sigma * (1 - sigma ** 2)
        return dsigma, d2sigma


if __name__ == '__main__':
    from hessQuik.tests import DerivativeCheckTestsActivationFunction
    torch.set_default_dtype(torch.float64)

    nex = 11  # no. of examples
    d = 4  # no. of input features

    x = torch.randn(nex, d)
    dx = torch.randn_like(x)

    f = tanhActivation()

    derivativeTests = DerivativeCheckTestsActivationFunction()

    print('======= FORWARD =======')
    derivativeTests.run_forward_hessian_test(f, x, dx, verbose=True)

    print('======= BACKWARD =======')
    derivativeTests.run_backward_hessian_test(f, x, dx, verbose=True)