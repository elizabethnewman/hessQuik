import torch
from hessQuik.activations import hessQuikActivationFunction


class antiTanhActivation(hessQuikActivationFunction):

    def __init__(self):
        super(antiTanhActivation, self).__init__()

    def forward(self, x, do_gradient=False, do_Hessian=False, do_Laplacian=False):
        (dsigma, d2sigma) = (None, None)

        # forward
        sigma = torch.abs(x) + torch.log(1 + torch.exp(-2.0 * torch.abs(x)))

        # compute derivatives
        if do_gradient or do_Hessian or do_Laplacian:
            if self.reverse_mode is not None:
                dsigma, d2sigma = self.compute_derivatives(x, do_Hessian=do_Hessian, do_Laplacian=do_Laplacian)
            else:
                self.ctx = (x,)

        return sigma, dsigma, d2sigma

    def compute_derivatives(self, *args, do_Hessian=False, do_Laplacian=False):
        x = args[0]
        dsigma = torch.tanh(x)
        d2sigma = None
        if do_Hessian or do_Laplacian:
            d2sigma = 1 - dsigma ** 2

        return dsigma, d2sigma


if __name__ == '__main__':
    from hessQuik.utils import input_derivative_check
    torch.set_default_dtype(torch.float64)

    nex = 11  # no. of examples
    d = 4  # no. of input features

    x = torch.randn(nex, d)

    f = antiTanhActivation()

    print('======= FORWARD =======')
    f.reverse_mode = False
    input_derivative_check(f, x, do_Hessian=True, verbose=True)

    print('======= BACKWARD =======')
    f.reverse_mode = True
    input_derivative_check(f, x, do_Hessian=True, verbose=True)

