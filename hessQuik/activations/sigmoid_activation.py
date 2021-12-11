import torch
from hessQuik.activations import activationFunction


class sigmoidActivation(activationFunction):

    def __init__(self):
        super(sigmoidActivation, self).__init__()

    def forward(self, x, do_gradient=False, do_Hessian=False, reverse_mode=False):
        (dsigma, d2sigma) = (None, None)

        sigma = torch.sigmoid(x)

        if reverse_mode:
            self.ctx = (sigma,)
        else:
            if do_gradient or do_Hessian:
                dsigma = sigma * (1 - sigma)
                if do_Hessian:
                    d2sigma = dsigma * (1 - 2 * sigma)

        return sigma, dsigma, d2sigma

    def backward(self, do_Hessian=False):
        sigma, = self.ctx
        d2sigma = None
        dsigma = sigma * (1 - sigma)
        if do_Hessian:
            d2sigma = dsigma * (1 - 2 * sigma)
        return dsigma, d2sigma


if __name__ == '__main__':
    from hessQuik.utils import input_derivative_check
    torch.set_default_dtype(torch.float64)

    nex = 11  # no. of examples
    d = 4  # no. of input features

    x = torch.randn(nex, d)

    f = sigmoidActivation()

    print('======= FORWARD =======')
    input_derivative_check(f, x, do_Hessian=True, verbose=True, reverse_mode=False)

    print('======= BACKWARD =======')
    input_derivative_check(f, x, do_Hessian=True, verbose=True, reverse_mode=True)
