import torch
import torch.nn.functional as F
from hessQuik.activations import hessQuikActivationFunction


class softplusActivation(hessQuikActivationFunction):
    r"""
    Applies the softplus activation function to each entry of the incoming data.

    Examples::

        >>> import hessQuik.activations as act
        >>> act_func = act.softplusActivation()
        >>> x = torch.randn(10, 4)
        >>> sigma, dsigma, d2sigma = act_func(x, do_gradient=True, do_Hessian=True)

    """

    def __init__(self, beta: float = 1.0, threshold: float = 20.0) -> None:
        r"""

        :param beta: parameter affecting steepness of the softplus function.  Default: 1.0
        :type beta: float
        :param threshold: parameter for numerical stability.  Uses identity function when :math:`\beta x > threshold`
        :type threshold: float
        """
        super(softplusActivation, self).__init__()
        self.beta = beta
        self.threshold = threshold

<<<<<<< HEAD
    def forward(self, x, do_gradient=False, do_Hessian=False, do_Laplacian=False):
=======
    def forward(self, x, do_gradient=False, do_Hessian=False, forward_mode=True):
        r"""
        Activates each entry of incoming data via

        .. math::

            \sigma(x)  = \frac{1}{\beta}\ln(1 + e^{\beta x})
        """

>>>>>>> c846faf2d50607569f3f073aa019d49e967371c4
        (dsigma, d2sigma) = (None, None)

        # forward propagate
        sigma = F.softplus(x, beta=self.beta, threshold=self.threshold)

        # compute derivatives
        if do_gradient or do_Hessian:
<<<<<<< HEAD
            if self.reverse_mode is not None:
                dsigma, d2sigma = self.compute_derivatives(x, do_Hessian=do_Hessian, do_Laplacian=do_Laplacian)
=======
            if forward_mode is not None:
                dsigma, d2sigma = self.compute_derivatives(x, do_Hessian=do_Hessian)
>>>>>>> c846faf2d50607569f3f073aa019d49e967371c4
            else:
                # backward mode, but do not compute yet
                self.ctx = (x,)

        return sigma, dsigma, d2sigma

<<<<<<< HEAD
    def compute_derivatives(self, *args, do_Hessian=False, do_Laplacian=False):
=======
    def compute_derivatives(self, *args, do_Hessian=False):
        r"""
        Computes the first and second derivatives of each entry of the incoming data via

        .. math::
            \begin{align}
                \sigma'(x)  &= \frac{1}{1 + e^{-\beta x}}\\
                \sigma''(x) &= \frac{\beta}{2\cosh(\beta x) + 2}
            \end{align}

        """
>>>>>>> c846faf2d50607569f3f073aa019d49e967371c4
        x = args[0]
        d2sigma = None

<<<<<<< HEAD
        if do_Hessian or do_Laplacian:
            d2sigma = torch.zeros_like(x)
            d2sigma[idx] = self.beta * torch.exp(self.beta * x[idx]) / ((1 + torch.exp(self.beta * x[idx])) ** 2)
=======
        dsigma = 1 / (1 + torch.exp(-self.beta * x))
        if do_Hessian:
            d2sigma = self.beta / (2 + 2 * torch.cosh(self.beta * x))

        # for numerical stability
        idx = (self.beta * x > self.threshold).nonzero(as_tuple=True)
        if len(idx[0]) > 0:
            dsigma[idx] = 1.0
            if do_Hessian:
                d2sigma[idx] = 0.0
>>>>>>> c846faf2d50607569f3f073aa019d49e967371c4

        return dsigma, d2sigma


if __name__ == '__main__':
    from hessQuik.utils import input_derivative_check
    torch.set_default_dtype(torch.float64)

    nex = 11  # no. of examples
    d = 4  # no. of input features

    x = 100 * torch.randn(nex, d)

    f = softplusActivation()

    print('======= FORWARD =======')
    input_derivative_check(f, x, do_Hessian=True, verbose=True, forward_mode=True)

    print('======= BACKWARD =======')
    input_derivative_check(f, x, do_Hessian=True, verbose=True, forward_mode=False)