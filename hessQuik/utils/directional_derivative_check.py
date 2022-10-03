import torch


def directional_derivative_check(f, x, k=3, tol=1e-8, verbose=False):
    """
    Compute
    :param f: hessQuik network
    :type f: hessQuik.NN
    :param x: input data
    :type x: torch.Tensor
    :param k: number of unit directions
    :type k: int
    :param tol: printouts
    :type tol: bool
    :param verbose: printouts
    :type verbose: bool
    :return:
    :rtype:
    """

    # obtain full gradient and Hessian
    f0, df0, d2f0 = f(x, do_gradient=True, do_Hessian=True)

    # input and output dimensions
    d = x.shape[1]
    m = f0.shape[1]

    # --------------------------------- #
    # FORWARD
    # --------------------------------- #
    # forward mode directional derivative in random direction
    v = torch.randn(k, d, dtype=x.dtype, device=x.device)
    fv, dfv, d2fv = f(x, do_gradient=True, do_Hessian=True, v=v, forward_mode=True)

    # apply gradient and Hessian in given direction
    vdf0 = v.unsqueeze(0) @ df0
    err_forward_grad = torch.norm(vdf0 - dfv).item()

    vd2f0v = (v @ d2f0.permute(0, 3, 1, 2) @ v.T).permute(0, 3, 2, 1)
    err_forward_hess = torch.norm(vd2f0v - d2fv).item()

    forward_mode_flag_grad = (err_forward_grad < tol)
    forward_mode_flag_hess = (err_forward_hess < tol)
    if verbose:
        print('FORWARD MODE: grad = %0.4e\t hess = %0.4e' % (err_forward_grad, err_forward_hess))
        if forward_mode_flag_grad:
            print("\t Gradient PASSED!")
        else:
            print("\t Gradient FAILED.")

        if forward_mode_flag_hess:
            print("\t Hessian PASSED!")
        else:
            print("\t Hessian FAILED.")

    # --------------------------------- #
    # BACKWARD
    # --------------------------------- #
    v2 = torch.randn(m, k, dtype=x.dtype, device=x.device)
    fv2, dfv2, d2fv2 = f(x, do_gradient=True, do_Hessian=True, v=v2, forward_mode=False)

    df0v2 = df0 @ v2
    err_backward_grad = torch.norm(df0v2 - dfv2).item()
    d2f0v2 = d2f0 @ v2
    err_backward_hess = torch.norm(d2f0v2 - d2fv2).item()

    backward_mode_flag_grad = (err_backward_grad < tol)
    backward_mode_flag_hess = (err_backward_hess < tol)
    if verbose:
        print('BACKWARD MODE: grad = %0.4e\t hess = %0.4e' % (err_backward_grad, err_backward_hess))
        if backward_mode_flag_grad:
            print("\t Gradient PASSED!")
        else:
            print("\t Gradient FAILED.")
        if backward_mode_flag_hess:
            print("\t Hessian PASSED!")
        else:
            print("\t Hessian FAILED.")

    return forward_mode_flag_grad, forward_mode_flag_hess, backward_mode_flag_grad, backward_mode_flag_hess


def directional_derivative_laplacian_check(f, x, k=3, tol=1e-8, verbose=False):
    # obtain full gradient and Hessian
    f0, df0, d2f0 = f(x, do_gradient=True, do_Hessian=True)

    # input and output dimensions
    d = x.shape[1]

    # forward mode directional derivative in random direction
    v = torch.randn(k, d, dtype=x.dtype, device=x.device)
    fv, dfv, d2fv = f(x, do_gradient=True, do_Laplacian=True, do_Hessian=False, v=v)

    # apply gradient and Hessian in given direction
    vdf0 = v.unsqueeze(0) @ df0
    err_forward_grad = torch.norm(vdf0 - dfv).item()

    vd2f0v = (v @ d2f0.permute(0, 3, 1, 2) @ v.T).permute(0, 3, 2, 1)
    idx = torch.arange(k)
    err_forward_lap = torch.norm(vd2f0v[:, idx, idx].sum(axis=1, keepdims=True) - d2fv).item()

    forward_mode_flag_grad = (err_forward_grad < tol)
    forward_mode_flag_lap = (err_forward_lap < tol)
    if verbose:
        print('FORWARD MODE: grad = %0.4e\t lap = %0.4e' % (err_forward_grad, err_forward_lap))
        if forward_mode_flag_grad:
            print("\t Gradient PASSED!")
        else:
            print("\t Gradient FAILED.")

        if forward_mode_flag_lap:
            print("\t Laplacian PASSED!")
        else:
            print("\t Laplacian FAILED.")

