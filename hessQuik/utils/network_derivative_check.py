import torch
from math import log2, floor
import hessQuik
from hessQuik.utils import convert_to_base, extract_data, insert_data


def network_derivative_check(f, x, do_Hessian=False, num_test=15, base=2, tol=0.1, verbose=False):

    loss_df, loss_d2f = 0.0, 0.0
    dg0, d2g0 = None, None

    # initial evaluation
    f0, df0, d2f0 = f(x, do_gradient=True, do_Hessian=do_Hessian)
    g0 = torch.randn_like(f0).detach()
    loss_f = 0.5 * torch.norm(f0 - g0) ** 2

    if df0 is not None:
        dg0 = torch.randn_like(df0).detach()
        loss_df = 0.5 * torch.norm(df0 - dg0) ** 2

    if d2f0 is not None:
        d2g0 = torch.randn_like(d2f0).detach()
        loss_d2f = 0.5 * torch.norm(d2f0 - d2g0) ** 2

    loss = loss_f + loss_df + loss_d2f
    loss.backward()

    loss0 = loss.detach()
    theta0 = extract_data(f, 'data')
    grad_theta0 = extract_data(f, 'grad')
    dtheta = torch.randn_like(theta0)
    dtheta = dtheta / torch.norm(dtheta)

    dfdtheta = (grad_theta0 * dtheta).sum()

    # ---------------------------------------------------------------------------------------------------------------- #
    # derivative check
    E0, E1 = [], []
    loss_dft, loss_d2ft = 0.0, 0.0
    for k in range(num_test):
        h = base ** (-k)
        insert_data(f, theta0 + h * dtheta)
        ft, dft, d2ft = f(x, do_gradient=True, do_Hessian=do_Hessian)

        loss_ft = 0.5 * torch.norm(ft - g0) ** 2

        if dg0 is not None:
            loss_dft = 0.5 * torch.norm(dft - dg0) ** 2

        if d2g0 is not None:
            loss_d2ft = 0.5 * torch.norm(d2ft - d2g0) ** 2

        losst = loss_ft + loss_dft + loss_d2ft

        E0.append((loss0 - losst).item())
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
    torch.set_default_dtype(torch.float32)

    nex = 11  # no. of examples
    d = 4  # no. of input features

    x = torch.randn(nex, d)
    dx = torch.randn_like(x)

    # f = net.NN(lay.singleLayer(d, 7, act=act.softplusActivation()),
    #            lay.singleLayer(7, 5, act=act.identityActivation()))

    width = 7
    f = net.NN(lay.singleLayer(d, width, act=act.tanhActivation()),
               net.resnetNN(width, 4, act=act.softplusActivation()),
               net.fullyConnectedNN([width, 13, 5], act=act.quadraticActivation()),
               lay.singleLayer(5, 3, act=act.identityActivation()),
               lay.quadraticLayer(3, 2)
               )

    network_derivative_check(f, x, do_Hessian=True, verbose=True)

