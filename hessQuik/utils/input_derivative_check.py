import torch
from math import log2, floor, ceil
import hessQuik
from hessQuik.utils import convert_to_base


def input_derivative_check(f, x, do_Hessian=False, num_test=15, base=2.0, tol=0.1, verbose=False):
    dx = torch.randn_like(x)
    curvx = None

    # initial evaluation
    if isinstance(f, hessQuik.activations.hessQuikActivationFunction):
        f0, df0, d2f0 = f(x, do_gradient=True, do_Hessian=do_Hessian)
        dfdx = df0 * dx

        if d2f0 is not None:
            curvx = torch.sum(dx.unsqueeze(0) * d2f0 * dx.unsqueeze(0), dim=0)

    else:
        f0, df0, d2f0, _ = f(x, do_gradient=True, do_Hessian=do_Hessian)
        dfdx = torch.matmul(df0.transpose(1, 2), dx.unsqueeze(2)).squeeze(2)

        if d2f0 is not None:
            curvx = torch.sum(dx.unsqueeze(2).unsqueeze(3) * d2f0 * dx.unsqueeze(1).unsqueeze(3), dim=(1, 2))

    # ---------------------------------------------------------------------------------------------------------------- #
    # derivative check
    grad_check, hess_check = None, None
    E0, E1, E2 = [], [], []
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


if __name__ == '__main__':
    from hessQuik.networks import NN
    from hessQuik.layers import singleLayer
    from hessQuik.activations import softplusActivation
    torch.set_default_dtype(torch.float32)

    nex = 11  # no. of examples
    d = 4  # no. of input features

    x = torch.randn(nex, d)
    dx = torch.randn_like(x)

    f = NN(singleLayer(d, 7, act=softplusActivation()),
           singleLayer(7, 5, act=softplusActivation()), reverse_mode=False)

    input_derivative_check(f, x, do_Hessian=True, verbose=True)

