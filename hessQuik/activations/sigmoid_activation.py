import torch
from hessQuik.activations import hessQuikActivationFunction


class sigmoidActivation(hessQuikActivationFunction):

    def __init__(self):
        super(sigmoidActivation, self).__init__()

    def forward(self, x, do_gradient=False, do_Hessian=False):
        (dsigma, d2sigma) = (None, None)

        # forward propagate
        sigma = torch.sigmoid(x)

        # compute derivatves
        if do_gradient or do_Hessian:
            if self.reverse_mode is not None:
                dsigma, d2sigma = self.compute_derivatives(sigma, do_Hessian=do_Hessian)
            else:
                self.ctx = (sigma,)

        return sigma, dsigma, d2sigma

    def compute_derivatives(self, *args, do_Hessian=False):
        sigma = args[0]
        dsigma = sigma * (1 - sigma)
        d2sigma = None
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
    f.reverse_mode = False
    input_derivative_check(f, x, do_Hessian=True, verbose=True)

    print('======= BACKWARD =======')
    f.reverse_mode = True
    input_derivative_check(f, x, do_Hessian=True, verbose=True)
