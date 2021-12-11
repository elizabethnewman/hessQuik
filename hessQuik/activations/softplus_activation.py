import torch
from hessQuik.activations import activationFunction


class softplusActivation(activationFunction):

    def __init__(self, beta=1, threshold=20):
        super(softplusActivation, self).__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, x, do_gradient=False, do_Hessian=False, reverse_mode=False):
        (dsigma, d2sigma) = (None, None)
        # implement ourselves!
        # sigma = F.softplus(x)
        # dsigma = torch.exp(x) / (1 + torch.exp(x))
        # d2sigma = torch.exp(x) / ((1 + torch.exp(x))**2)
        if reverse_mode:
            self.ctx = (x,)

        # initialize sigma
        sigma = torch.clone(x)

        # find values of x below threshold
        idx = self.beta * x < self.threshold
        sigma[idx] = (1 / self.beta) * torch.log(1 + torch.exp(self.beta * x[idx]))

        if do_gradient or do_Hessian:
            idx = self.beta * x < self.threshold
            dsigma = torch.ones_like(x)
            dsigma[idx] = torch.exp(self.beta * x[idx]) / (1 + torch.exp(self.beta * x[idx]))

            if do_Hessian:
                d2sigma = torch.zeros_like(x)
                d2sigma[idx] = self.beta * torch.exp(self.beta * x[idx]) / ((1 + torch.exp(self.beta * x[idx])) ** 2)

        return sigma, dsigma, d2sigma

    def backward(self, do_Hessian=False):
        x, = self.ctx
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
    input_derivative_check(f, x, do_Hessian=True, verbose=True, reverse_mode=False)

    print('======= BACKWARD =======')
    input_derivative_check(f, x, do_Hessian=True, verbose=True, reverse_mode=True)