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

    def forward(self, x, do_gradient=False, do_Hessian=False, forward_mode=True):
        r"""
        Activates each entry of incoming data via

        .. math::

            \sigma(x)  = \frac{1}{\beta}\ln(1 + e^{\beta x})
        """

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
        r"""
        Computes the first and second derivatives of each entry of the incoming data via

        .. math::
            \begin{align}
                \sigma'(x)  &= \frac{1}{1 + e^{-\beta x}}\\
                \sigma''(x) &= \frac{\beta}{2\cosh(\beta x) + 2}
            \end{align}

        """
        x = args[0]
        d2sigma = None

        dsigma = torch.zeros_like(x)
        idl = self.beta * x < -self.threshold  # input too small
        idr = self.beta * x > self.threshold  # input too large
        idok = self.beta * torch.abs(x) <= self.threshold # input in range

        dsigma[idr] = 1.0
        dsigma[idl] = 0.0
        dsigma[idok] = 1 / (1 + torch.exp(-self.beta * x[idok]))

        if do_Hessian:
            d2sigma = torch.zeros_like(x)
            d2sigma[idok] = self.beta / (2 + 2 * torch.cosh(self.beta * x[idok]))
            d2sigma[idr] = 0
            d2sigma[idl] = 0

        return dsigma, d2sigma


if __name__ == '__main__':
    from hessQuik.utils import input_derivative_check
    torch.set_default_dtype(torch.float64)

    nex = 11  # no. of examples
    d = 4  # no. of input features

    x = 1 * torch.randn(nex, d)

    f = softplusActivation()

    print('======= FORWARD =======')
    input_derivative_check(f, x, do_Hessian=True, verbose=True, forward_mode=True)

    print('======= BACKWARD =======')
    input_derivative_check(f, x, do_Hessian=True, verbose=True, forward_mode=False)

    x.requires_grad=True
    g,dg,d2g = f(1000*x, do_gradient=True, do_Hessian=True)
    fun = torch.sum(g)+torch.sum(dg)+torch.sum(d2g)
    fun.backward()
    print(fun)
    print(x.grad)
