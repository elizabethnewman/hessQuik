import torch
from math import log2, floor, ceil
import hessQuik
from hessQuik.utils import convert_to_base


def input_derivative_check(f, x, do_Hessian=False, forward_mode=True, num_test=15, base=2.0, tol=0.1, verbose=False):

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
    # check if order is 2 at least half of the time
    eps = torch.finfo(x.dtype).eps
    grad_check = (sum((torch.log2(E1[:-1] / E1[1:]) / log2(base)) > (2 - tol)) > 3)
    grad_check = (grad_check or (torch.kthvalue(E1, num_test // 3)[0] < (100 * eps)))

    if curvx is not None:
        hess_check = (sum(torch.log2(E2[:-1] / E2[1:]) / log2(base) > (3 - tol)) > 3)
        hess_check = (hess_check or (torch.kthvalue(E2, num_test // 3)[0] < (100 * eps)))

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


def input_derivative_check_finite_difference(f, x, do_Hessian=False, forward_mode=True,
                                             eps=1e-4, atol=1e-5, rtol=1e-3, verbose=False):

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

