import torch
from hessQuik.activations import hessQuikActivationFunction


class identityActivation(hessQuikActivationFunction):
    r"""
    Applies the identity activation function to each entry of the incoming data.

    Examples::

        >>> import hessQuik.activations as act
        >>> act_func = act.identityActivation()
        >>> x = torch.randn(10, 4)
        >>> sigma, dsigma, d2sigma = act_func(x, do_gradient=True, do_Hessian=True)

    """

    def __init__(self):
        super(identityActivation, self).__init__()

    def forward(self, x, do_gradient=False, do_Hessian=False, forward_mode=True):
        r"""
        Activates each entry of incoming data via

        .. math::

            \sigma(x)  = x
        """
        (dsigma, d2sigma) = (None, None)

        # forward propagate
        sigma = x

        # compute derivatives
        if do_gradient or do_Hessian:
            if forward_mode is not None:
                dsigma, d2sigma = self.compute_derivatives(x, do_Hessian=do_Hessian)
            else:
                self.ctx = (x,)

        return sigma, dsigma, d2sigma

    def compute_derivatives(self, *args, do_Hessian=False):
        r"""
        Computes the first and second derivatives of each entry of the incoming data via

        .. math::
            \begin{align}
                \sigma'(x)  &= 1\\
                \sigma''(x) &= 0
            \end{align}

        """
        x = args[0]
        d2sigma = None
        dsigma = torch.ones_like(x)

        if do_Hessian:
            d2sigma = torch.zeros_like(x)

        return dsigma, d2sigma


if __name__ == '__main__':
    from hessQuik.utils import input_derivative_check
    torch.set_default_dtype(torch.float64)

    nex = 11  # no. of examples
    d = 4  # no. of input features

    x = torch.randn(nex, d)

    f = identityActivation()

    print('======= FORWARD =======')
    input_derivative_check(f, x, do_Hessian=True, verbose=True, forward_mode=True)

    print('======= BACKWARD =======')
    input_derivative_check(f, x, do_Hessian=True, verbose=True, forward_mode=False)

