import torch
from math import log2, floor
import hessQuik
from hessQuik.utils import convert_to_base, extract_data, insert_data
from typing import Callable, Tuple, Optional, Union


def network_derivative_check(f: torch.nn.Module, x: torch.Tensor, do_Hessian: bool = False,
                             forward_mode: bool = True, num_test: int = 15, base: float = 2.0, tol: float = 0.1,
                             verbose: bool = False) -> Optional[bool]:
    r"""
    Taylor approximation test to verify derivatives.  Form the approximation by perturbing the
    network weights :math:`\theta` in the direction :math:`p` with step size :math:`h > 0` via

    .. math::

        \Phi(\theta + h p) \approx \Phi(\theta) + h\nabla_{\theta} \Phi(\theta)^\top p

    where :math:`\Phi` is the objective function and :math:`\theta` are the network weights.  This test uses the loss

    .. math::

        \Phi(\theta) = \frac{1}{2}\|f_{\theta}(x)\|^2 + \frac{1}{2}\|\nabla f_{\theta}(x)\|^2 +  \frac{1}{2}\|\nabla^2 f_{\theta}(x)\|^2

    to validate network gradient computation after computing derivatives of the
    input features of the network :math:`f_{\theta}`.

    As :math:`h \downarrow 0^+`, the error between the approximation and the true value will decrease.
    The rate of decrease indicates the accuracy of the derivative computation.
    For details, see Chapter 5 of `Computational Methods for Electromagnetics`_ by Eldad Haber.

    .. _Computational Methods for Electromagnetics: https://epubs.siam.org/doi/book/10.1137/1.9781611973808

    Examples::

        >>> import hessQuik.activations as act, hessQuik.layers as lay, hessQuik.networks as net
        >>> torch.set_default_dtype(torch.float64)  # use double precision to check implementations
        >>> x = torch.randn(10, 4)
        >>> width, depth = 8, 3
        >>> f = net.NN(lay.singleLayer(4, width, act=act.tanhActivation()), net.resnetNN(width, depth, h=1.0, act=act.tanhActivation()), lay.singleLayer(width, 1, act=act.identityActivation()))
        >>> network_derivative_check(f, x, do_Hessian=True, verbose=True, forward_mode=True)
            h                           E0                      E1
            1.00 x 2^(00)		1.97 x 2^(02)		1.05 x 2^(01)
            1.00 x 2^(-01)		1.14 x 2^(02)		1.71 x 2^(-02)
            1.00 x 2^(-02)		1.20 x 2^(01)		1.50 x 2^(-04)
            1.00 x 2^(-03)		1.22 x 2^(00)		1.40 x 2^(-06)
            1.00 x 2^(-04)		1.24 x 2^(-01)		1.35 x 2^(-08)
            1.00 x 2^(-05)		1.24 x 2^(-02)		1.32 x 2^(-10)
            1.00 x 2^(-06)		1.24 x 2^(-03)		1.31 x 2^(-12)
            1.00 x 2^(-07)		1.24 x 2^(-04)		1.30 x 2^(-14)
            1.00 x 2^(-08)		1.25 x 2^(-05)		1.30 x 2^(-16)
            1.00 x 2^(-09)		1.25 x 2^(-06)		1.30 x 2^(-18)
            1.00 x 2^(-10)		1.25 x 2^(-07)		1.30 x 2^(-20)
            1.00 x 2^(-11)		1.25 x 2^(-08)		1.30 x 2^(-22)
            1.00 x 2^(-12)		1.25 x 2^(-09)		1.30 x 2^(-24)
            1.00 x 2^(-13)		1.25 x 2^(-10)		1.30 x 2^(-26)
            1.00 x 2^(-14)		1.25 x 2^(-11)		1.30 x 2^(-28)
            Gradient PASSED!

    :param f: callable function that returns value, gradient, and Hessian
    :type f: torch.nn.Module
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

    """
    loss_df, loss_d2f = 0.0, 0.0

    # initial evaluation
    f0, df0, d2f0 = f(x, do_gradient=True, do_Hessian=do_Hessian, forward_mode=forward_mode)

    # compute loss
    loss_f = 0.5 * torch.norm(f0) ** 2

    if df0 is not None:
        loss_df = 0.5 * torch.norm(df0) ** 2

    if d2f0 is not None:
        loss_d2f = 0.5 * torch.norm(d2f0) ** 2

    loss = loss_f + loss_df + loss_d2f
    loss.backward()

    loss0 = loss.detach()
    theta0 = extract_data(f, 'data')
    grad_theta0 = extract_data(f, 'grad')

    # perturbation
    dtheta = torch.randn_like(theta0)
    dtheta = dtheta / torch.norm(dtheta)

    # directional derivative
    dfdtheta = (grad_theta0 * dtheta).sum()

    # ---------------------------------------------------------------------------------------------------------------- #
    # derivative check
    if verbose:
        headers = ('h', 'E0', 'E1')
        print(('{:<20s}' * len(headers)).format(*headers))

    with torch.no_grad():
        E0, E1 = [], []
        loss_dft, loss_d2ft = 0.0, 0.0
        for k in range(num_test):
            h = base ** (-k)
            insert_data(f, theta0 + h * dtheta)
            ft, dft, d2ft = f(x, do_gradient=True, do_Hessian=do_Hessian)

            # compute loss
            loss_ft = 0.5 * torch.norm(ft) ** 2

            if df0 is not None:
                loss_dft = 0.5 * torch.norm(dft) ** 2

            if d2f0 is not None:
                loss_d2ft = 0.5 * torch.norm(d2ft) ** 2

            losst = loss_ft + loss_dft + loss_d2ft
            E0.append(torch.norm(loss0 - losst).item())
            E1.append(torch.norm(loss0 + h * dfdtheta - losst).item())

            printouts = convert_to_base((E0[-1], E1[-1]))

            if verbose:
                print(((1 + len(printouts) // 2) * '%0.2f x 2^(%0.2d)\t\t') % ((1, -k) + printouts))

    E0, E1 = torch.tensor(E0), torch.tensor(E1)

    # ---------------------------------------------------------------------------------------------------------------- #
    # check if order is 2 at least half of the time
    eps = torch.finfo(x.dtype).eps
    grad_check = (sum((torch.log2(E1[:-1] / E1[1:]) / log2(base)) > (2 - tol)) > 3)
    grad_check = (grad_check or (torch.kthvalue(E1, num_test // 3)[0] < (100 * eps)))

    if verbose:
        if grad_check:
            print('Gradient PASSED!')
        else:
            print('Gradient FAILED.')

    return grad_check


if __name__ == '__main__':
    import hessQuik.networks as net
    import hessQuik.layers as lay
    import hessQuik.activations as act
    torch.set_default_dtype(torch.float64)

    nex = 11  # no. of examples
    d = 2  # no. of input features

    x = torch.randn(nex, d)
    dx = torch.randn_like(x)

    # f = net.NN(lay.singleLayer(d, 7, act=act.softplusActivation()),
    #            lay.singleLayer(7, 5, act=act.identityActivation()))

    width = 8
    depth = 8
    f = net.NN(lay.singleLayer(d, width, act=act.tanhActivation()),
               net.resnetNN(width, depth, h=1.0, act=act.tanhActivation()),
               lay.singleLayer(width, 1, act=act.identityActivation()))

    # width = 7
    # f = net.NN(lay.singleLayer(d, width, act=act.tanhActivation()),
    #            net.resnetNN(width, 4, act=act.softplusActivation()),
    #            net.fullyConnectedNN([width, 13, 5], act=act.quadraticActivation()),
    #            lay.singleLayer(5, 3, act=act.identityActivation()),
    #            lay.quadraticLayer(3, 2)
    #            )

    network_derivative_check(f, x, do_Hessian=True, verbose=True)

