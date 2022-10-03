import torch
from typing import Callable, Tuple, Optional


def input_derivative_check_finite_difference_laplacian(f: Callable, x: torch.Tensor,
                                             do_Laplacian: bool = False,
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
        >>> input_derivative_check_finite_difference_laplacian(f, x, do_Hessian=True, verbose=True, forward_mode=True)
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
    f0, df0, lapf0 = f(x, do_gradient=True, do_Laplacian=do_Laplacian)

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
    lap_check = None
    if do_Laplacian:

        lapf0_approx = torch.zeros(df0.shape[0], df0.shape[1], df0.shape[2])
        for i in range(d):
            ei = torch.zeros(d, device=f0.device, dtype=f0.dtype)
            ei[i] = eps
            f1, *_ = f(x + (ei + ei).unsqueeze(0))
            f4, *_ = f(x - (ei + ei).unsqueeze(0))

            lapf0_approx[:, i] += (f1 - 2 * f0 + f4) / (4 * (eps ** 2))

        lapf0_approx = lapf0_approx.sum(dim=1, keepdim=True)
        err = torch.norm(lapf0 - lapf0_approx).item()
        rel_err = err / max(torch.norm(lapf0).item(), eps)

        lap_check = (err < atol and rel_err < rtol)
        if verbose:
            print('Laplacian Finite Difference: Error = %0.4e, Relative Error = %0.4e' % (err, rel_err))

            if lap_check:
                print('Laplacian PASSED!')
            else:
                print('Laplacian FAILED.')

    return grad_check, lap_check
