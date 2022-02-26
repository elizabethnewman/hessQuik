import torch
from hessQuik.activations import hessQuikActivationFunction


class sigmoidActivation(hessQuikActivationFunction):
    r"""
    Sigmoid function

    .. math::

        \begin{align}
            \sigma(x)   &= \frac{1}{1 + e^{-x}}\\
            \sigma'(x)  &= \sigma(x)(1 - \sigma(x))\\
            \sigma''(x) &= \sigma'(x)(1 - 2 * \sigma(x))
        \end{align}

    """

    def __init__(self):
        super(sigmoidActivation, self).__init__()

    def forward(self, x, do_gradient=False, do_Hessian=False, forward_mode=True):
        """
        :meta private:
        """
        (dsigma, d2sigma) = (None, None)

        # forward propagate
        sigma = torch.sigmoid(x)

        # compute derivatves
        if do_gradient or do_Hessian:
            if forward_mode is not None:
                dsigma, d2sigma = self.compute_derivatives(sigma, do_Hessian=do_Hessian)
            else:
                self.ctx = (sigma,)

        return sigma, dsigma, d2sigma

    def compute_derivatives(self, *args, do_Hessian=False):
        """
        :meta private:
        """
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
    input_derivative_check(f, x, do_Hessian=True, verbose=True, forward_mode=True)

    print('======= BACKWARD =======')
    input_derivative_check(f, x, do_Hessian=True, verbose=True, forward_mode=False)
