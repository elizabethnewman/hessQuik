import torch
import torch.nn.functional as F
from hessQuik.activations import hessQuikActivationFunction


class softplusActivation(hessQuikActivationFunction):

    def __init__(self, beta=1, threshold=20):
        super(softplusActivation, self).__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, x, do_gradient=False, do_Hessian=False, forward_mode=True):
        (dsigma, d2sigma) = (None, None)

        # forward propagate
        sigma = F.softplus(x, beta=self.beta, threshold=self.threshold)

        # compute derivatives
        if do_gradient or do_Hessian:
            if forward_mode is not None:
                dsigma, d2sigma = self.compute_derivatives(x, do_Hessian=do_Hessian)
            else:
                # backward mode, but do not compute yet
                self.ctx = (x,)

        return sigma, dsigma, d2sigma

    def compute_derivatives(self, *args, do_Hessian=False):
        x = args[0]
        d2sigma = None
        idx = self.beta * x < self.threshold
        dsigma = torch.ones_like(x)
        dsigma[idx] = torch.exp(self.beta * x[idx]) / (1 + torch.exp(self.beta * x[idx]))

        if do_Hessian:
            d2sigma = torch.zeros_like(x)
            d2sigma[idx] = self.beta * torch.exp(self.beta * x[idx]) / ((1 + torch.exp(self.beta * x[idx])) ** 2)

        return dsigma, d2sigma


if __name__ == '__main__':
    from hessQuik.utils import input_derivative_check
    torch.set_default_dtype(torch.float64)

    nex = 11  # no. of examples
    d = 4  # no. of input features

    x = torch.randn(nex, d)

    f = softplusActivation()

    print('======= FORWARD =======')
    input_derivative_check(f, x, do_Hessian=True, verbose=True, forward_mode=True)

    print('======= BACKWARD =======')
    input_derivative_check(f, x, do_Hessian=True, verbose=True, forward_mode=False)