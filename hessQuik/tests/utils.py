import torch
from math import log2, floor
from hessQuik.utils import input_derivative_check, convert_to_base


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
