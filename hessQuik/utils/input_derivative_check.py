import torch
from math import log2, floor, ceil
import hessQuik
from hessQuik.utils import convert_to_base
from typing import Callable, Tuple, Optional, Union
import math

def input_derivative_check(f: Union[torch.nn.Module, Callable], x: torch.Tensor, do_Hessian: bool = False,
                           forward_mode: bool = True, num_test: int = 15, base: float = 2.0, tol: float = 0.1,
                           verbose: float = False) -> Tuple[Optional[bool], Optional[bool]]:
    r"""
    Taylor approximation test to verify derivatives.  Form the approximation by perturbing the input :math:`x`
    in the direction :math:`p` with step size :math:`h > 0` via

    .. math::

        f(x + h p) \approx f(x) + h\nabla f(x)^\top p + \frac{1}{2}p^\top \nabla^2f(x) p

    As :math:`h \downarrow 0^+`, the error between the approximation and the true value will decrease.
    The rate of decrease indicates the accuracy of the derivative computation.
    For details, see Chapter 5 of `Computational Methods for Electromagnetics`_ by Eldad Haber.

    .. _Computational Methods for Electromagnetics: https://epubs.siam.org/doi/book/10.1137/1.9781611973808

    Examples::

        >>> from hessQuik.layers import singleLayer
        >>> torch.set_default_dtype(torch.float64)  # use double precision to check implementations
        >>> x = torch.randn(10, 4)
        >>> f = singleLayer(4, 7, act=act.softplusActivation())
        >>> input_derivative_check(f, x, do_Hessian=True, verbose=True, forward_mode=True)
            h                           E0                      E1                      E2
            1.00 x 2^(00)		1.62 x 2^(-02)		1.70 x 2^(-07)		1.02 x 2^(-12)
            1.00 x 2^(-01)		1.63 x 2^(-03)		1.70 x 2^(-09)		1.06 x 2^(-15)
            1.00 x 2^(-02)		1.63 x 2^(-04)		1.69 x 2^(-11)		1.08 x 2^(-18)
            1.00 x 2^(-03)		1.63 x 2^(-05)		1.69 x 2^(-13)		1.09 x 2^(-21)
            1.00 x 2^(-04)		1.63 x 2^(-06)		1.69 x 2^(-15)		1.09 x 2^(-24)
            1.00 x 2^(-05)		1.63 x 2^(-07)		1.69 x 2^(-17)		1.10 x 2^(-27)
            1.00 x 2^(-06)		1.63 x 2^(-08)		1.69 x 2^(-19)		1.10 x 2^(-30)
            1.00 x 2^(-07)		1.63 x 2^(-09)		1.69 x 2^(-21)		1.10 x 2^(-33)
            1.00 x 2^(-08)		1.63 x 2^(-10)		1.69 x 2^(-23)		1.10 x 2^(-36)
            1.00 x 2^(-09)		1.63 x 2^(-11)		1.69 x 2^(-25)		1.10 x 2^(-39)
            1.00 x 2^(-10)		1.63 x 2^(-12)		1.69 x 2^(-27)		1.10 x 2^(-42)
            1.00 x 2^(-11)		1.63 x 2^(-13)		1.69 x 2^(-29)		1.10 x 2^(-45)
            1.00 x 2^(-12)		1.63 x 2^(-14)		1.69 x 2^(-31)		1.15 x 2^(-48)
            1.00 x 2^(-13)		1.63 x 2^(-15)		1.69 x 2^(-33)		1.33 x 2^(-50)
            1.00 x 2^(-14)		1.63 x 2^(-16)		1.69 x 2^(-35)		1.70 x 2^(-51)
            Gradient PASSED!
            Hessian PASSED!

    :param f: callable function that returns value, gradient, and Hessian
    :type f: torch.nn.Module or Callable
    :param x: input data
    :type x: torch.Tensor
    :param do_Hessian: If set to ``True``, the Hessian will be computed during the forward call. Default: ``False``
    :type do_Hessian: bool, optional
    :param forward_mode:  If set to ``False``, the derivatives will be computed in backward mode. Default: ``True``
    :type forward_mode: bool, optional
    :param num_test: number of perturbations
    :type num_test: int
    :param base: step size :math:`h = base^k`
    :type base: float
    :param tol: small tolerance to account for numerical errors when computing the order of approximation
    :type tol: float
    :param verbose: printout flag
    :type verbose: bool
    :return:
            - **grad_check** (*bool*) - if ``True``, gradient check passes
            - **hess_check** (*bool, optional*) -  if ``True``, Hessian check passes

    """
    # initial evaluation
    f0, df0, d2f0 = f(x, do_gradient=True, do_Hessian=do_Hessian, forward_mode=forward_mode)

    # ---------------------------------------------------------------------------------------------------------------- #
    # directional derivatives
    dx = torch.randn_like(x)
    dx = dx / torch.norm(x)
    curvx = None
    if isinstance(f, hessQuik.activations.hessQuikActivationFunction):
        dfdx = df0 * dx

        if d2f0 is not None:
            curvx = torch.sum(dx.unsqueeze(0) * d2f0 * dx.unsqueeze(0), dim=0)
    else:
        dfdx = torch.matmul(df0.transpose(1, 2), dx.unsqueeze(2)).squeeze(2)

        if d2f0 is not None:
            curvx = torch.sum(dx.unsqueeze(2).unsqueeze(3) * d2f0 * dx.unsqueeze(1).unsqueeze(3), dim=(1, 2))

    # ---------------------------------------------------------------------------------------------------------------- #
    # derivative check
    grad_check, hess_check = None, None
    E0, E1, E2 = [], [], []

    if verbose:
        headers = ('h', 'E0', 'E1')
        if do_Hessian:
            headers += ('E2',)
        print(('{:<20s}' * len(headers)).format(*headers))

    for k in range(num_test):
        h = base ** (-k)
        ft, *_ = f(x + h * dx, do_gradient=False, do_Hessian=False)

        E0.append(torch.norm(f0 - ft).item())
        E1.append(torch.norm(f0 + h * dfdx - ft).item())
        printouts = convert_to_base((E0[-1], E1[-1]))

        if curvx is not None:
            E2.append(torch.norm(f0 + h * dfdx + 0.5 * (h ** 2) * curvx - ft).item())
            printouts += convert_to_base((E2[-1],))

        if verbose:
            print(((1 + len(printouts) // 2) * '%0.2f x 2^(%0.2d)\t\t') % ((1, -k) + printouts))

    E0, E1, E2 = torch.tensor(E0), torch.tensor(E1), torch.tensor(E2)

    # ---------------------------------------------------------------------------------------------------------------- #
    # check if order is 2 enough of the time
    eps = torch.finfo(x.dtype).eps
    grad_check = (sum((torch.log2(E1[:-1] / E1[1:]) / log2(base)) > (2 - tol)) > num_test // 3)
    grad_check = (grad_check or (torch.kthvalue(E1, num_test // 4)[0] < (100 * eps)))

    if curvx is not None:
        hess_check = (sum((torch.log2(E2[:-1] / E2[1:]) / log2(base)) > (3 - tol)) > num_test // 3)
        hess_check = (hess_check or (torch.kthvalue(E2, num_test // 4)[0] < (100 * eps)))

    if verbose:
        if grad_check:
            print('Gradient PASSED!')
        else:
            print('Gradient FAILED.')

        if curvx is not None:
            if hess_check:
                print('Hessian PASSED!')
            else:
                print('Hessian FAILED.')

    return grad_check, hess_check


def input_derivative_check_finite_difference(f: Callable, x: torch.Tensor,
                                             do_Hessian: bool = False, forward_mode: bool = True,
                                             eps: float = 1e-4, atol: float = 1e-5, rtol: float = 1e-3,
                                             verbose: bool = False) -> Tuple[Optional[bool], Optional[bool]]:
    r"""
    Finite difference test to verify derivatives.  Form the approximation by perturbing each entry in the input
    in the unit direction with step size :math:`\varepsilon > 0`:

    .. math::

        \widetilde{\nabla f}_{i} = \frac{f(x_i + \varepsilon) -  f(x_i - \varepsilon)}{2\varepsilon}

    where :math:`x_i \pm \varepsilon` means add or subtract :math:`\varepsilon` from
    the i-th entry of the input :math:`x`, but leave the other entries unchanged.
    The notation :math:`\widetilde{(\cdot)}` indicates the finite difference approximation.

    Examples::

        >>> from hessQuik.layers import singleLayer
        >>> torch.set_default_dtype(torch.float64)  # use double precision to check implementations
        >>> x = torch.randn(10, 4)
        >>> f = singleLayer(4, 7, act=act.tanhActivation())
        >>> input_derivative_check_finite_difference(f, x, do_Hessian=True, verbose=True, forward_mode=True)
            Gradient Finite Difference: Error = 8.1720e-10, Relative Error = 2.5602e-10
            Gradient PASSED!
            Hessian Finite Difference: Error = 4.5324e-08, Relative Error = 4.4598e-08
            Hessian PASSED!

    :param f: callable function that returns value, gradient, and Hessian
    :type f: Callable
    :param x: input data
    :type x: torch.Tensor
    :param do_Hessian: If set to ``True``, the Hessian will be computed during the forward call. Default: ``False``
    :type do_Hessian: bool, optional
    :param forward_mode:  If set to ``False``, the derivatives will be computed in backward mode. Default: ``True``
    :type forward_mode: bool, optional
    :param eps: step size. Default: 1e-4
    :type eps: float
    :param atol: absolute tolerance, e.g., :math:`\|\nabla f - \widetilde{\nabla f}\| < atol`. Default: 1e-5
    :type atol: float
    :param rtol: relative tolerance, e.g., :math:`\|\nabla f - \widetilde{\nabla f}\|/\|\nabla f\| < rtol`. Default: 1e-3
    :type rtol: float
    :param verbose: printout flag
    :type verbose: bool
    :return:
            - **grad_check** (*bool*) - if ``True``, gradient check passes
            - **hess_check** (*bool, optional*) -  if ``True``, Hessian check passes
    """

    # compute initial gradient
    f0, df0, d2f0 = f(x, do_gradient=True, do_Hessian=do_Hessian, forward_mode=forward_mode)

    d = x.shape[1]

    # ---------------------------------------------------------------------------------------------------------------- #
    # test gradient
    df0_approx = torch.zeros_like(df0)
    for i in range(d):
        # perturbation in standard directions
        ei = torch.zeros(d, device=f0.device, dtype=f0.dtype)
        ei[i] = eps

        f_pos, *_ = f(x + ei.unsqueeze(0))
        f_neg, *_ = f(x - ei.unsqueeze(0))

        df0_approx[:, i] = (f_pos - f_neg) / (2 * eps)

    err = torch.norm(df0 - df0_approx).item()
    rel_err = err / torch.norm(df0).item()

    grad_check = (err < atol and rel_err < rtol)
    if verbose:
        print('Gradient Finite Difference: Error = %0.4e, Relative Error = %0.4e' % (err, rel_err))

        if grad_check:
            print('Gradient PASSED!')
        else:
            print('Gradient FAILED.')

    # ---------------------------------------------------------------------------------------------------------------- #
    # test Hessian
    # https://v8doc.sas.com/sashtml/ormp/chap5/sect28.htm
    hess_check = None
    if do_Hessian:

        d2f0_approx = torch.zeros_like(d2f0)
        for i in range(d):
            ei = torch.zeros(d, device=f0.device, dtype=f0.dtype)
            ei[i] = eps

            for j in range(d):
                ej = torch.zeros(d, device=f0.device, dtype=f0.dtype)
                ej[j] = eps

                f1, *_ = f(x + (ei + ej).unsqueeze(0))
                f2, *_ = f(x + (ei - ej).unsqueeze(0))
                f3, *_ = f(x + (-ei + ej).unsqueeze(0))
                f4, *_ = f(x - (ei + ej).unsqueeze(0))

                d2f0_approx[:, i, j] = (f1 - f2 - f3 + f4) / (4 * (eps ** 2))

        err = torch.norm(d2f0 - d2f0_approx).item()
        rel_err = err / max(torch.norm(d2f0).item(), eps)

        hess_check = (err < atol and rel_err < rtol)
        if verbose:
            print('Hessian Finite Difference: Error = %0.4e, Relative Error = %0.4e' % (err, rel_err))

            if hess_check:
                print('Hessian PASSED!')
            else:
                print('Hessian FAILED.')

    return grad_check, hess_check


if __name__ == '__main__':
    from hessQuik.networks import NN
    from hessQuik.layers import singleLayer
    import hessQuik.activations as act
    torch.set_default_dtype(torch.float64)

    nex = 11  # no. of examples
    d = 4  # no. of input features

    x = torch.randn(nex, d)
    dx = torch.randn_like(x)

    f = NN(singleLayer(d, 7, act=act.identityActivation()),
           singleLayer(7, 5, act=act.identityActivation()))

    input_derivative_check(f, x, do_Hessian=True, verbose=True)

    input_derivative_check_finite_difference(f, x, do_Hessian=True, verbose=True, forward_mode=False)

