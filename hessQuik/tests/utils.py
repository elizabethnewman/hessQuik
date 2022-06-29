from hessQuik.utils import input_derivative_check, input_derivative_check_finite_difference_laplacian, directional_derivative_laplacian_check, directional_derivative_check


def run_forward_gradient_test(f, x, **kwargs):
    grad_check, hess_check = input_derivative_check(f, x, do_Hessian=False, forward_mode=True, **kwargs)

    if grad_check:
        out = 'PASSED'
    else:
        out = 'FAILED'
    print('\t', out, ': forward_gradient_test')
    assert grad_check


def run_forward_hessian_test(f, x, **kwargs):
    grad_check, hess_check = input_derivative_check(f, x, do_Hessian=True, forward_mode=True, **kwargs)

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


def run_backward_gradient_test(f, x, **kwargs):
    grad_check, hess_check = input_derivative_check(f, x, do_Hessian=False, forward_mode=False, **kwargs)

    if grad_check:
        out = 'PASSED'
    else:
        out = 'FAILED'
    print('\t', out, ': backward_gradient_test')
    assert grad_check


def run_backward_hessian_test(f, x, **kwargs):
    grad_check, hess_check = input_derivative_check(f, x, do_Hessian=True, forward_mode=False, **kwargs)

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


def run_all_tests(f, x, num_test=15, base=2.0, tol=0.1, verbose=False):
    run_forward_gradient_test(f, x, num_test=num_test, base=base, tol=tol, verbose=verbose)
    run_forward_hessian_test(f, x, num_test=num_test, base=base, tol=tol, verbose=verbose)
    run_backward_gradient_test(f, x, num_test=num_test, base=base, tol=tol, verbose=verbose)
    run_backward_hessian_test(f, x, num_test=num_test, base=base, tol=tol, verbose=True)


def run_forward_gradient_test_finite_difference(f, x, **kwargs):
    grad_check, hess_check = input_derivative_check_finite_difference_laplacian(f, x, do_Laplacian=False, **kwargs)

    if grad_check:
        out = 'PASSED'
    else:
        out = 'FAILED'
    print('\t', out, ': forward_gradient_test')
    assert grad_check


def run_forward_laplacian_test(f, x, **kwargs):
    grad_check, lap_check = input_derivative_check_finite_difference_laplacian(f, x, do_Laplacian=True, **kwargs)

    if grad_check:
        out = 'PASSED'
    else:
        out = 'FAILED'
    print('\t', out, ': forward_laplacian_test gradient')
    assert grad_check

    if lap_check:
        out = 'PASSED'
    else:
        out = 'FAILED'
    print('\t', out, ': forward_laplacian_test Laplacian')
    assert lap_check


def run_all_tests_laplacian(f, x, eps=1e-4, atol=1e-5, rtol=1e-3, verbose=False):
    run_forward_gradient_test_finite_difference(f, x, eps=eps, atol=atol, rtol=rtol, verbose=verbose)
    run_forward_laplacian_test(f, x, eps=eps, atol=atol, rtol=rtol, verbose=verbose)


def run_all_tests_directional(f, x, k=3, tol=1e-10, verbose=False):
    directional_derivative_check(f, x, k=k, tol=tol, verbose=verbose)
    directional_derivative_laplacian_check(f, x, k=k, tol=tol, verbose=verbose)

