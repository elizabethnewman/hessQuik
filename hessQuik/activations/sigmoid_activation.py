import torch
from hessQuik.activations import hessQuikActivationFunction


class sigmoidActivation(hessQuikActivationFunction):
    r"""
    Applies the sigmoid activation function to each entry of the incoming data.

    Examples::

        >>> import hessQuik.activations as act
        >>> act_func = act.sigmoidActivation()
        >>> x = torch.randn(10, 4)
        >>> sigma, dsigma, d2sigma = act_func(x, do_gradient=True, do_Hessian=True)

    """

    def __init__(self):
        super(sigmoidActivation, self).__init__()

<<<<<<< HEAD
    def forward(self, x, do_gradient=False, do_Hessian=False, do_Laplacian=False):
=======
    def forward(self, x, do_gradient=False, do_Hessian=False, forward_mode=True):
        r"""
        Activates each entry of incoming data via

        .. math::

            \sigma(x)  = \frac{1}{1 + e^{-x}}
        """
>>>>>>> c846faf2d50607569f3f073aa019d49e967371c4
        (dsigma, d2sigma) = (None, None)

        # forward propagate
        sigma = torch.sigmoid(x)

        # compute derivatves
        if do_gradient or do_Hessian:
<<<<<<< HEAD
            if self.reverse_mode is not None:
                dsigma, d2sigma = self.compute_derivatives(sigma, do_Hessian=do_Hessian, do_Laplacian=do_Laplacian)
=======
            if forward_mode is not None:
                dsigma, d2sigma = self.compute_derivatives(sigma, do_Hessian=do_Hessian)
>>>>>>> c846faf2d50607569f3f073aa019d49e967371c4
            else:
                self.ctx = (sigma,)

        return sigma, dsigma, d2sigma

<<<<<<< HEAD
    def compute_derivatives(self, *args, do_Hessian=False, do_Laplacian=False):
=======
    def compute_derivatives(self, *args, do_Hessian=False):
        r"""
        Computes the first and second derivatives of each entry of the incoming data via

        .. math::
            \begin{align}
                \sigma'(x)  &= \sigma(x)(1 - \sigma(x))\\
                \sigma''(x) &= \sigma'(x)(1 - 2 * \sigma(x))
            \end{align}

        """
>>>>>>> c846faf2d50607569f3f073aa019d49e967371c4
        sigma = args[0]
        dsigma = sigma * (1 - sigma)
        d2sigma = None
        if do_Hessian or do_Laplacian:
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
