import torch


def laplcian_check_using_hessian(f, x, tol=1e-4):
    f0, df0, d2f0, _ = f(x, do_gradient=True, do_Hessian=True, do_Laplacian=False)

    _, _, _, lap_f = f(x, do_gradient=True, do_Hessian=False, do_Laplacian=True)

    err = torch.norm(d2f0[:, torch.arange(d2f0.shape[1]), torch.arange(d2f0.shape[1]), :].sum(1) - lap_f)
    rel_err = err / torch.norm(d2f0)

    if rel_err < tol:
        print('Laplcian PASSED! : rel. err = %0.4e' % rel_err)
    else:
        print('Laplcian FAILED. : rel. err = %0.4e' % rel_err)
