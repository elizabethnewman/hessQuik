import torch
from math import log2, floor
import hessQuik
from hessQuik.utils import convert_to_base, extract_data, insert_data


def network_derivative_check(f, x, do_Hessian=False, num_test=15, base=2, tol=0.1, verbose=False):

    # initial evaluation
    f0, df0, d2f0 = f(x, do_gradient=True, do_Hessian=do_Hessian)
    g0 = torch.randn_like(f0).detach()
    dg0 = torch.randn_like(df0).detach()
    d2g0 = torch.randn_like(d2f0).detach()

    loss = 0.5 * torch.norm(f0 - g0) ** 2 + 0.5 * torch.norm(df0 - dg0) ** 2 + 0.5 * torch.norm(d2f0 - d2g0) ** 2
    loss.backward()

    loss0 = loss.detach()
    theta0 = extract_data(f, 'data')
    grad_theta0 = extract_data(f, 'grad')
    dtheta = torch.randn_like(theta0)

    dfdtheta = (grad_theta0 * dtheta).sum()

    # ---------------------------------------------------------------------------------------------------------------- #
    # derivative check
    E0, E1 = [], []
    for k in range(num_test):
        h = base ** (-k)
        insert_data(f, theta0 + h * dtheta)
        ft, dft, d2ft = f(x, do_gradient=True, do_Hessian=do_Hessian)

        losst = 0.5 * torch.norm(ft - g0) ** 2 + 0.5 * torch.norm(dft - dg0) ** 2 + 0.5 * torch.norm(d2ft - d2g0) ** 2
        E0.append((loss0 - losst).item())
        E1.append(torch.norm(loss0 + h * dfdtheta - losst).item())
        printouts = convert_to_base((E0[-1], E1[-1]))

        if verbose:
            print(((1 + len(printouts) // 2) * '%0.2f x 2^(%0.2d)\t\t') % ((1, -k) + printouts))

    E0, E1 = torch.tensor(E0), torch.tensor(E1)

    # ---------------------------------------------------------------------------------------------------------------- #
    # check if order is 2 at least half of the time
    grad_check = sum((torch.log2(E1[:-1] / E1[1:]) / log2(base)) > (2 - tol)) > num_test / 2
    grad_check = (grad_check or sum(E1) / len(E1) < 1e-8)

    if verbose:
        if grad_check:
            print('Gradient PASSED!')
        else:
            print('Gradient FAILED.')

    return grad_check


if __name__ == '__main__':
    from hessQuik.networks import NN
    from hessQuik.layers import singleLayer
    from hessQuik.activations import softplusActivation
    torch.set_default_dtype(torch.float64)

    nex = 11  # no. of examples
    d = 4  # no. of input features

    x = torch.randn(nex, d)
    dx = torch.randn_like(x)

    f = NN(singleLayer(d, 7, act=softplusActivation()),
           singleLayer(7, 5, act=softplusActivation()))

    network_derivative_check(f, x, do_Hessian=True, verbose=True)
