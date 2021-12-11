import torch
from math import log2, floor
from hessQuik.utils import input_derivative_check, convert_to_base


def derivative_check(f, x, dx, f0, dfx, curvx=None, num_test=15, base=2, tol=0.1, verbose=False):

    (E0, E1, E2) = (torch.zeros(num_test), torch.zeros(num_test), torch.zeros(num_test))
    for k in range(num_test):
        h = base ** (-k)
        ft, *_ = f(x + h * dx, do_gradient=False, do_Hessian=False)

        E0[k] = torch.norm(f0 - ft)
        E1[k] = torch.norm(f0 + h * dfx - ft)

        if curvx is not None:
            E2[k] = torch.norm(f0 + h * dfx + 0.5 * (h ** 2) * curvx - ft)

            if verbose:
                printouts = convert_to_base((E0[k], E1[k], E2[k]), base)
                print((4 * '%0.2f x 2^(%0.2d)\t\t') % ((1, -k) + printouts))

        else:
            if verbose:
                printouts = convert_to_base((E0[k], E1[k]), base)
                print((3 * '%0.2f x 2^(%0.2d)\t\t') % ((1, -k) + printouts))

    # check if order is 2 at least half of the time
    grad_check = sum((torch.log2(E1[:-1] / E1[1:]) / log2(base)) > (2 - tol)) > num_test / 2
    grad_check = (grad_check or sum(E1) / len(E1) < 1e-8)
    # assert (grad_check or sum(E1) / len(E1) < 1e-8), "Gradient failed"

    if verbose:
        if grad_check:
            print('Gradient PASSED!')
        else:
            print('Gradient FAILED.')

        if curvx is not None:
            # check if order is 3 at least half the time or very small
            hess_check = sum((torch.log2(E2[:-1] / E2[1:]) / log2(base)) > (3 - tol)) > num_test / 2
            hess_check = (hess_check or sum(E2) / len(E1) < 1e-8)

            if hess_check:
                print('Hessian PASSED!')
            else:
                print('Hessian FAILED.')

    if curvx is None:
        return grad_check
    else:
        # check if order is 3 at least half the time or very small
        hess_check = sum((torch.log2(E2[:-1] / E2[1:]) / log2(base)) > (3 - tol)) > num_test / 2
        hess_check = (hess_check or sum(E2) / len(E1) < 1e-8)
        # assert (hess_check or sum(E2) / len(E1) < 1e-8), "Hessian failed"
        return grad_check, hess_check


def run_forward_gradient_test(f, x, num_test=15, base=2, tol=0.1, verbose=False):
    grad_check, hess_check = input_derivative_check(f, x, do_Hessian=False, reverse_mode=False,
                                                    num_test=num_test, base=base, tol=tol, verbose=verbose)

    if grad_check:
        out = 'PASSED'
    else:
        out = 'FAILED'
    print('\t', out, ': forward_gradient_test')
    assert grad_check


def run_forward_hessian_test(f, x, num_test=15, base=2, tol=0.1, verbose=False):
    grad_check, hess_check = input_derivative_check(f, x, do_Hessian=True, reverse_mode=False,
                                                    num_test=num_test, base=base, tol=tol, verbose=verbose)

    if grad_check:
        out = 'PASSED'
    else:
        out = 'FAILED'
    print('\t', out, ': forward_hessian_test gradient')
    assert grad_check

    if hess_check:
        out = 'PASSED'
    else:
        out = 'FAILED'
    print('\t', out, ': forward_hessian_test Hessian')
    assert hess_check


def run_backward_gradient_test(f, x, num_test=15, base=2, tol=0.1, verbose=False):
    grad_check, hess_check = input_derivative_check(f, x, do_Hessian=False, reverse_mode=True,
                                                    num_test=num_test, base=base, tol=tol, verbose=verbose)

    if grad_check:
        out = 'PASSED'
    else:
        out = 'FAILED'
    print('\t', out, ': backward_gradient_test')
    assert grad_check


def run_backward_hessian_test(f, x, num_test=15, base=2, tol=0.1, verbose=False):
    grad_check, hess_check = input_derivative_check(f, x, do_Hessian=True, reverse_mode=True,
                                                    num_test=num_test, base=base, tol=tol, verbose=verbose)

    if grad_check:
        out = 'PASSED'
    else:
        out = 'FAILED'
    print('\t', out, ': backward_hessian_test gradient')
    assert grad_check

    if hess_check:
        out = 'PASSED'
    else:
        out = 'FAILED'
    print('\t', out, ': backward_hessian_test Hessian')
    assert hess_check


def run_all_tests(f, x, num_test=15, base=2, tol=0.1, verbose=False):
    run_forward_gradient_test(f, x, num_test=num_test, base=base, tol=tol, verbose=verbose)
    run_forward_hessian_test(f, x, num_test=num_test, base=base, tol=tol, verbose=verbose)
    run_backward_gradient_test(f, x, num_test=num_test, base=base, tol=tol, verbose=verbose)
    run_backward_hessian_test(f, x, num_test=num_test, base=base, tol=tol, verbose=verbose)
