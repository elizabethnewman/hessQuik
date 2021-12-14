import torch
from hessQuik.activations import activationFunction


class tanhActivation(activationFunction):

    def __init__(self):
        super(tanhActivation, self).__init__()

    def forward(self, x, do_gradient=False, do_Hessian=False, reverse_mode=False):
        (dsigma, d2sigma) = (None, None)

        # forward propagate
        sigma = torch.tanh(x)

        # compute derivatives
        if do_gradient or do_Hessian:
            if self.reverse_mode is not None:
                dsigma, d2sigma = self.compute_derivatives(sigma, do_Hessian=do_Hessian)
            else:
                self.ctx = (sigma,)

        return sigma, dsigma, d2sigma

    def compute_derivatives(self, *args, do_Hessian=False):
        sigma = args[0]
        d2sigma = None
        dsigma = 1 - sigma ** 2
        if do_Hessian:
            d2sigma = -2 * sigma * (1 - sigma ** 2)

        return dsigma, d2sigma


if __name__ == '__main__':
    from hessQuik.utils import input_derivative_check
    torch.set_default_dtype(torch.float64)

    nex = 11  # no. of examples
    d = 4  # no. of input features

    x = torch.randn(nex, d)

    f = tanhActivation()

    print('======= FORWARD =======')
    f.reverse_mode = False
    input_derivative_check(f, x, do_Hessian=True, verbose=True)

    print('======= BACKWARD =======')
    f.reverse_mode = True
    input_derivative_check(f, x, do_Hessian=True, verbose=True)
