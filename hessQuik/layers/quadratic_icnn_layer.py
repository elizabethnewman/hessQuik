import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from hessQuik.layers import hessQuikLayer
from typing import Union, Tuple


class quadraticICNNLayer(hessQuikLayer):
    r"""
    Evaluate and compute derivatives of a ICNN quadratic layer.

    Examples::

        >>> import hessQuik.layers as lay
        >>> f = lay.quadraticICNNLayer(4, None, 2)
        >>> x = torch.randn(10, 4)
        >>> fx, dfdx, d2fd2x = f(x, do_gradient=True, do_Hessian=True)
        >>> print(fx.shape, dfdx.shape, d2fd2x.shape)
        torch.Size([10, 1]) torch.Size([10, 4, 1]) torch.Size([10, 4, 4, 1])

    """

    def __init__(self, input_dim: int, in_features: Union[int, None], rank: int, device=None, dtype=None) -> None:
        r"""

        :param input_dim: dimension of network inputs
        :type input_dim: int
        :param in_features: number of input features, :math:`n_{in}`.  For only ICNN quadratic layer, set ``in_features = None``
        :type in_features: int or ``None``
        :param rank: number of columns of quadratic matrix, :math:`r`.  In practice, :math:`r < n_{in}`
        :type rank: int
        :var v: weight vector for network inputs of size :math:`(d,)`
        :var w: weight vector for input features of size :math:`(n_{in},)`
        :var A: weight matrix for quadratic term of size :math:`(d, r)`
        :var mu: additive scalar bias
        :var nonneg: pointwise function to force :math:`l` to have nonnegative weights. Default ``torch.nn.functional.softplus``
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(quadraticICNNLayer, self).__init__()

        self.input_dim = input_dim
        self.in_features = in_features
        self.rank = rank
        self.ctx = None
        self.nonneg = F.softplus

        # create final layer
        if in_features is not None:
            self.w = nn.Parameter(torch.empty(in_features, **factory_kwargs))
        else:
            self.register_parameter('w', None)

        self.v = nn.Parameter(torch.empty(input_dim, **factory_kwargs))
        self.mu = nn.Parameter(torch.empty(1, **factory_kwargs))
        self.A = nn.Parameter(torch.empty(rank, input_dim, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:

        if self.in_features is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.w, a=-bound, b=bound)
        else:
            bound = 1 / math.sqrt(self.input_dim)

        nn.init.uniform_(self.v, a=-bound, b=bound)
        nn.init.uniform_(self.mu)
        bound = 1 / math.sqrt(self.input_dim)
        nn.init.uniform_(self.A, a=-bound, b=bound)

    def dim_input(self) -> int:
        r"""
        number of input features + dimension of network inputs
        """
        return self.in_features + self.input_dim

    def dim_output(self) -> int:
        r"""
        scalar
        """
        return 1

    def forward(self, ux: torch.Tensor, do_gradient: bool = False, do_Hessian: bool = False, forward_mode: bool = True,
                dudx: Union[torch.Tensor, None] = None, d2ud2x: Union[torch.Tensor, None] = None) \
            -> Tuple[torch.Tensor, Union[torch.Tensor, None], Union[torch.Tensor, None]]:
        r"""
        Forward propagation through ICNN layer of the form, for one sample :math:`n_s = 1`,

        .. math::

            f(x) =
            \left[\begin{array}{c}u(x) & x\end{array}\right]
            \left[\begin{array}{c}w^+ \\ v\end{array}\right] + \frac{1}{2} x  A A^\top  x^\top + \mu

        Here, :math:`u(x)` is the input into the layer of size :math:`(n_s, n_{in})` which is
        a function of the input of the network, :math:`x` of size :math:`(n_s, d)`.
        The output features, :math:`f(x)`, are of size :math:`(n_s, 1)`.
        The notation :math:`(\cdot)^+` is a function that makes the weights of a matrix nonnegative.

        As an example, for one sample, :math:`n_s = 1`, the gradient with respect to :math:`x` is of the form

        .. math::

                \nabla_x f = \left[\begin{array}{c}(w^+)^\top & v^\top\end{array}\right]
                \left[\begin{array}{c} \nabla_x u \\ I\end{array}\right] + x A A^\top

        where :math:`I` is the :math:`d \times d` identity matrix.

        """
        (df, d2f) = (None, None)
        AtA = self.A.t() @ self.A

        if self.w is None:
            w = torch.empty(0, dtype=self.v.dtype, device=self.v.device)
        else:
            w = self.nonneg(self.w)

        wv = torch.cat((w, self.v), dim=0)
        x = ux[:, -self.input_dim:]

        # forward propagate
        f = ux @ wv + 0.5 * torch.sum((x @ AtA) * x, dim=1) + self.mu

        # ------------------------------------------------------------------------------------------------------------ #
        if (do_gradient or do_Hessian) and forward_mode is True:
            if self.in_features is None:
                z = torch.empty(ux.shape[0], 0)
            else:
                z = torch.zeros(ux.shape[0], self.in_features)

            df = wv.unsqueeze(0) + torch.cat((z, x @ AtA), dim=1)

            # -------------------------------------------------------------------------------------------------------- #
            if do_Hessian:
                d2f = AtA

                if d2ud2x is not None:
                    d2f = dudx[:, :, -self.input_dim:] @ d2f @ dudx[:, :, -self.input_dim:].permute(0, 2, 1)
                    z = torch.zeros(x.shape[0], self.in_features)
                    d2f += (d2ud2x @ (torch.cat((w, self.v), dim=0).unsqueeze(0)
                                      + torch.cat((z, x @ AtA), dim=1)).unsqueeze(1).unsqueeze(-1)).squeeze()

                d2f = d2f.unsqueeze(-1)
                if d2f.ndim < 4:
                    e = torch.ones(x.shape[0], device=x.device, dtype=x.dtype).view(-1, 1, 1, 1)
                    d2f = e * d2f.unsqueeze(0)

            # -------------------------------------------------------------------------------------------------------- #
            # finish computing gradient
            if dudx is not None:
                df = (dudx @ df.unsqueeze(-1)).squeeze()

            df = df.unsqueeze(-1)

        if (do_gradient or do_Hessian) and forward_mode is not True:
            self.ctx = (ux,)
            if forward_mode is False:
                df, d2f = self.backward(do_Hessian=do_Hessian)

        return f.unsqueeze(-1), df, d2f

    def backward(self, do_Hessian: bool = False,
                 dgdf: Union[torch.Tensor, None] = None, d2gd2f: Union[torch.Tensor, None] = None) \
            -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        r"""
        Backward propagation through quadratic ICNN layer of the form, for one sample :math:`n_s = 1`,

        .. math::

                f\left(\begin{bmatrix} u & x \end{bmatrix}\right) =\left[\begin{array}{c}u & x\end{array}\right]
                \left[\begin{array}{c}w^+ \\ v\end{array}\right] + \frac{1}{2} x  A A^\top x^\top + \mu

        Here, the network is :math:`g` is a function of :math:`f(u)`.

        The gradient of the layer with respect to :math:`\begin{bmatrix} u & x \end{bmatrix}` is of the form

        .. math::

            \nabla_{[u,x]} f = \begin{bmatrix}(w^+)^\top & v^\top + x A A^\top\end{bmatrix}.

        """
        d2f = None

        ux = self.ctx[0]
        x = ux[:, -self.input_dim:]
        AtA = self.A.t() @ self.A  # TODO: recompute this or store it?

        wv = self.v
        if self.w is not None:
            wv = torch.cat((self.nonneg(self.w), wv), dim=0)

        z = torch.empty(ux.shape[0], 0)
        if self.in_features is not None:
            z = torch.zeros(ux.shape[0], self.in_features)

        df = wv.unsqueeze(0) + torch.cat((z, x @ AtA), dim=1)

        if do_Hessian:
            e = torch.ones(x.shape[0], 1, 1, dtype=AtA.dtype, device=AtA.device)
            d2f = torch.zeros(x.shape[0], ux.shape[1], ux.shape[1])
            d2f[:, -self.input_dim:, -self.input_dim:] = e * AtA
            d2f = d2f.unsqueeze(-1)

        return df.unsqueeze(-1), d2f


if __name__ == '__main__':
    from hessQuik.utils import input_derivative_check
    torch.set_default_dtype(torch.float64)

    nex = 11  # no. of examples
    d = 4  # no. of input features
    in_feat = 5
    m = 13  # rank
    x = torch.randn(nex, d)
    f = quadraticICNNLayer(d, None, m)

    print('======= FORWARD =======')
    input_derivative_check(f, x, do_Hessian=True, verbose=True, forward_mode=True)

    print('======= BACKWARD =======')
    input_derivative_check(f, x, do_Hessian=True, verbose=True, forward_mode=False)