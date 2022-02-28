import torch
from torch import Tensor
from typing import Tuple, Optional


def peaks(y: Tensor, do_gradient: bool = False, do_Hessian: bool = False) -> \
        Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    r"""
    Generate data from the `MATLAB 2D peaks function`_

    .. _MATLAB 2D peaks function: https://www.mathworks.com/help/matlab/ref/peaks.html

    Examples::

         import matplotlib.pyplot as plt
         from matplotlib import cm
         from mpl_toolkits import mplot3d
         x, y = torch.linspace(-3, 3, 100), torch.linspace(-3, 3, 100)
         grid_x, grid_y = torch.meshgrid(x, y)
         grid_xy = torch.concat((grid_x.reshape(-1, 1), grid_y.reshape(-1, 1)), dim=1)
         grid_z, *_ = peaks(grid_xy)
         fig = plt.figure()
         ax = plt.axes(projection='3d')
         surf = ax.plot_surface(grid_x, grid_y, grid_z.reshape(grid_x.shape), cmap=cm.viridis)
         ax.set_xlabel('x')
         ax.set_ylabel('y')
         ax.set_zlabel('z')
         plt.show()

    :param y: (x, y) coordinates with shape :math:`(n_s, 2)` where :math:`n_s` is the number of samples
    :type y: torch.Tensor
    :param do_gradient: If set to ``True``, the gradient will be computed during the forward call. Default: ``False``
    :type do_gradient: bool, optional
    :param do_Hessian: If set to ``True``, the Hessian will be computed during the forward call. Default: ``False``
    :type do_Hessian: bool, optional
    :return:
            - **f** (*torch.Tensor*) - value of peaks function at each coordinate with shape :math:`(n_s, 1)`
            - **dfdx** (*torch.Tensor* or ``None``) - value of gradient at each coordinate  with shape :math:`(n_s, 2)`
            - **d2fd2x** (*torch.Tensor* or ``None``) - value of Hessian at each coordinate with shape :math:`(n_s, 4)`

    """

    df, d2f = None, None

    # function
    e1 = torch.exp(-1.0 * (y[:, 0]**2 + (y[:, 1] + 1)**2))
    f1 = 3 * (1 - y[:, 0])**2 * e1

    e2 = torch.exp(-y[:, 0]**2 - y[:, 1]**2)
    f2 = -10 * (y[:, 0] / 5 - y[:, 0]**3 - y[:, 1]**5) * e2

    e3 = torch.exp(-(y[:, 0] + 1)**2 - y[:, 1]**2)
    f3 = -(1 / 3) * e3
    f = f1 + f2 + f3

    if do_gradient or do_Hessian:
        # gradient
        dexp1x = -2 * y[:, 0]
        dexp1y = -2 * (y[:, 1] + 1)

        df1x = -6 * (1 - y[:, 0]) * e1 + dexp1x * f1
        df1y = dexp1y * f1

        dexp2x = -2 * y[:, 0]
        dexp2y = -2 * y[:, 1]
        df2x = -10 * (1 / 5 - 3 * y[:, 0]**2) * e2 + dexp2x * f2
        df2y = 50 * y[:, 1]**4 * e2 + dexp2y * f2

        dexp3x = -2 * (y[:, 0] + 1)
        dexp3y = -2 * y[:, 1]
        df3x = dexp3x * f3
        df3y = dexp3y * f3

        dfx = (df1x + df2x + df3x).unsqueeze(-1)
        dfy = (df1y + df2y + df3y).unsqueeze(- 1)

        df = torch.cat((dfx, dfy), dim=1)
        df = df.unsqueeze(-1)

        if do_Hessian:
            # Hessian
            dexp1xx = -2
            dexp1yy = -2
            d2f1xx = 6 * e1 + -6 * (1 - y[:, 0]) * e1 * dexp1x + dexp1xx * f1 + dexp1x * df1x
            d2f1xy = dexp1y * df1x
            d2f1yy = dexp1yy * f1 + dexp1y * df1y

            dexp2xx = -2
            dexp2yy = -2
            d2f2xx = 60 * y[:, 0] * e2 + -10 * (1 / 5 - 3 * y[:, 0] ** 2) * e2 * dexp2x + dexp2xx * f2 + dexp2x * df2x
            d2f2xy = 50 * y[:, 1] ** 4 * e2 * dexp2x + dexp2y * df2x
            d2f2yy = 200 * y[:, 1] ** 3 * e2 + 50 * y[:, 1] ** 4 * e2 * dexp2y + dexp2yy * f2 + dexp2y * df2y

            dexp3xx = -2
            dexp3yy = -2
            d2f3xx = dexp3x * df3x + dexp3xx * f3
            d2f3xy = dexp3x * df3y
            d2f3yy = dexp3y * df3y + dexp3yy * f3

            d2fxx = (d2f1xx + d2f2xx + d2f3xx).unsqueeze(-1)
            d2fxy = (d2f1xy + d2f2xy + d2f3xy).unsqueeze(-1)
            d2fyy = (d2f1yy + d2f2yy + d2f3yy).unsqueeze(-1)

            d2f = torch.cat((torch.cat((d2fxx, d2fxy), dim=1).unsqueeze(-1),
                             torch.cat((d2fxy, d2fyy), dim=1).unsqueeze(-1)), dim=-1)

            d2f = d2f.unsqueeze(-1)

    return f.unsqueeze(-1), df, d2f


if __name__ == "__main__":
    from hessQuik.tests import derivative_check
    torch.set_default_dtype(torch.float64)

    x = -3 + 6 * torch.rand(11, 2)

    f0, df0, d2f0 = peaks(x, do_gradient=True, do_Hessian=True)

    # directional derivative
    dx = torch.randn_like(x)
    dfx = torch.matmul(df0.transpose(1, 2), dx.unsqueeze(2)).squeeze(2)

    curvx = None
    if d2f0 is not None:
        curvx = torch.sum(dx.unsqueeze(2).unsqueeze(3) * d2f0 * dx.unsqueeze(1).unsqueeze(3), dim=(1, 2))

    derivative_check(peaks, x, dx, f0, dfx, curvx, verbose=True)
