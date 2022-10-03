import torch
from hessQuik.activations import hessQuikActivationFunction


class tanhActivation(hessQuikActivationFunction):
    r"""
    Applies the hyperbolic tangent activation function to each entry of the incoming data.

    Examples::

        >>> import hessQuik.activations as act
        >>> act_func = act.tanhActivation()
        >>> x = torch.randn(10, 4)
        >>> sigma, dsigma, d2sigma = act_func(x, do_gradient=True, do_Hessian=True)

    """

    def __init__(self):
        super(tanhActivation, self).__init__()

    def forward(self, x, do_gradient=False, do_Hessian=False, forward_mode=True):
        r"""
        Activates each entry of incoming data via

        .. math::

            \sigma(x)  = \tanh(x)
        """
        (dsigma, d2sigma) = (None, None)

        # forward propagate
        sigma = torch.tanh(x)

        # compute derivatives
        if do_gradient or do_Hessian:
            if forward_mode is not None:
                dsigma, d2sigma = self.compute_derivatives(sigma, do_Hessian=do_Hessian)
            else:
                self.ctx = (sigma,)

        return sigma, dsigma, d2sigma

    def compute_derivatives(self, *args, do_Hessian=False):
        r"""
        Computes the first and second derivatives of each entry of the incoming data via

        .. math::
            \begin{align}
                \sigma'(x)  &= 1 - \tanh^2(x)\\
                \sigma''(x) &= -2\tanh(x) (1 - \tanh^2(x))
            \end{align}

        """
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
    input_derivative_check(f, x, do_Hessian=True, verbose=True, forward_mode=True)

    print('======= BACKWARD =======')
    input_derivative_check(f, x, do_Hessian=True, verbose=True, forward_mode=False)
