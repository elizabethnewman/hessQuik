import torch
import torch.nn as nn
import math
from hessQuik.layers import hessQuikLayer


class quadraticLayer(hessQuikLayer):
    r"""
    Evaluate and compute derivatives of a ICNN quadratic layer.

    Examples::

        >>> import hessQuik.layers as lay
        >>> f = lay.quadraticLayer(4, 2)
        >>> x = torch.randn(10, 4)
        >>> fx, dfdx, d2fd2x = f(x, do_gradient=True, do_Hessian=True)
        >>> print(fx.shape, dfdx.shape, d2fd2x.shape)
        torch.Size([10, 1]) torch.Size([10, 4, 1]) torch.Size([10, 4, 4, 1])

    """
    """
    f(x) = x @ v + 0.5 * x.t() @ A.t() @ A @ x + mu
    """

    def __init__(self, in_features: int, rank: int, device=None, dtype=None) -> None:
        r"""
        :param in_features: number of input features, :math:`n_{in}`
        :type in_features: int
        :param rank: number of columns of quadratic matrix, :math:`r`.  In practice, :math:`r < n_{in}`
        :type rank: int
        :var v: weight vector for network inputs of size :math:`(d,)`
        :var A: weight matrix for quadratic term of size :math:`(d, r)`
        :var mu: additive scalar bias
        """

        factory_kwargs = {'device': device, 'dtype': dtype}
        super(quadraticLayer, self).__init__()

        self.in_features = in_features
        self.rank = rank
        self.ctx = None

        # create final layer
        self.v = nn.Parameter(torch.empty(self.in_features, **factory_kwargs))
        self.mu = nn.Parameter(torch.empty(1, **factory_kwargs))
        self.A = nn.Parameter(torch.empty(self.rank, self.in_features, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.v, a=-bound, b=bound)
        nn.init.uniform_(self.mu)
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.A, a=-bound, b=bound)

    def dim_input(self) -> int:
        """
        number of input features
        """
        return self.in_features

    def dim_output(self) -> int:
        """
        scalar
        """
        return 1

    def forward(self, u, do_gradient=False, do_Hessian=False, forward_mode=True, dudx=None, d2ud2x=None):
        r"""
        Forward propagation through quadratic layer of the form, for one sample :math:`n_s = 1`,

        .. math::

            f(x) = u(x) v + \frac{1}{2} u(x)  A A^\top  u(x)^\top + \mu

        Here, :math:`u(x)` is the input into the layer of size :math:`(n_s, n_{in})` which is
        a function of the input of the network, :math:`x`.
        The output features, :math:`f(x)`, are of size :math:`(n_s, 1)`.

        The gradient with respect to :math:`x` is of the form

        .. math::

                \nabla_x f = (v^\top + u A A^\top) \nabla_x u

        """
        (df, d2f) = (None, None)
        AtA = self.A.t() @ self.A
        f = u @ self.v + 0.5 * torch.sum((u @ AtA) * u, dim=1) + self.mu

        if (do_gradient or do_Hessian) and forward_mode is True:
            df = self.v.unsqueeze(0) + u @ AtA

            if do_Hessian:
                d2f = AtA

                if d2ud2x is not None:
                    d2f = dudx @ d2f @ dudx.permute(0, 2, 1)
                    d2f += (d2ud2x @ df.unsqueeze(1).unsqueeze(-1)).squeeze()

                d2f = d2f.unsqueeze(-1)

            # finish computing gradient
            df = df.unsqueeze(-1)

            if dudx is not None:
                df = dudx @ df

        if (do_gradient or do_Hessian) and forward_mode is not True:
            self.ctx = (u,)
            if forward_mode is False:
                df, d2f = self.backward(do_Hessian=do_Hessian)

        return f.unsqueeze(-1), df, d2f

    def backward(self, do_Hessian=False, dgdf=None, d2gd2f=None):
        r"""
        Backward propagation through quadratic ICNN layer of the form, for one sample :math:`n_s = 1`,

        .. math::

                f(u) = u v + \frac{1}{2} u A A^\top  u^\top + \mu

        Here, the network is :math:`g` is a function of :math:`f(u)`.

        The gradient of the layer with respect to :math:`u` is of the form

        .. math::

            \nabla_u f = v^\top + u A A^\top.

        """
        d2f = None

        x = self.ctx[0]
        AtA = self.A.t() @ self.A  # TODO: recompute this or store it?
        df = self.v.unsqueeze(0) + torch.matmul(x, AtA)

        if do_Hessian:
            # TODO: improve wasteful storage
            d2f = (torch.ones(x.shape[0], 1, 1, dtype=AtA.dtype, device=AtA.device) * AtA).unsqueeze(-1)

        return df.unsqueeze(-1), d2f


if __name__ == '__main__':
    from hessQuik.utils import input_derivative_check
    torch.set_default_dtype(torch.float64)

    # problem setup
    nex = 11  # no. of examples
    d = 4  # no. of input dimensiona features
    m = 7  # rank
    x = torch.randn(nex, d)
    f = quadraticLayer(d, m)

    print('======= FORWARD =======')
    input_derivative_check(f, x, do_Hessian=True, verbose=True, forward_mode=True)

    print('======= BACKWARD =======')
    input_derivative_check(f, x, do_Hessian=True, verbose=True, forward_mode=False)


