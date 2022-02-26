import torch
from hessQuik.activations import hessQuikActivationFunction


class antiTanhActivation(hessQuikActivationFunction):
    r"""
    Antiderivative of hyperbolic tangent

    .. math::

        \begin{align}
            \sigma(x)   &= \ln|\cosh(x)|\\
            \sigma'(x)  &= \tanh(x)\\
            \sigma''(x) &= 1 - \tanh^2(x)
        \end{align}

    """

    def __init__(self):
        super(antiTanhActivation, self).__init__()

    def forward(self, x, do_gradient=False, do_Hessian=False, forward_mode=True):
        """
        :meta private:
        """

        (dsigma, d2sigma) = (None, None)

        # forward
        sigma = torch.abs(x) + torch.log(1 + torch.exp(-2.0 * torch.abs(x)))

        # compute derivatives
        if do_gradient or do_Hessian:
            if forward_mode is not None:
                dsigma, d2sigma = self.compute_derivatives(x, do_Hessian=do_Hessian)
            else:
                self.ctx = (x,)

        return sigma, dsigma, d2sigma

    def compute_derivatives(self, *args, do_Hessian=False):
        """
        :meta private:
        """
        x = args[0]
        dsigma = torch.tanh(x)
        d2sigma = None
        if do_Hessian:
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
    input_derivative_check(f, x, do_Hessian=True, verbose=True, forward_mode=True)

    print('======= BACKWARD =======')
    input_derivative_check(f, x, do_Hessian=True, verbose=True, forward_mode=False)

