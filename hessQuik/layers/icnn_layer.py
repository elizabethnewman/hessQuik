import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from hessQuik.layers import hessQuikLayer
import hessQuik.activations as act
from typing import Union, Tuple


class ICNNLayer(hessQuikLayer):
    r"""
    Evaluate and compute derivatives of a single layer.

    Examples::

        >>> import torch, hessQuik.layers as lay
        >>> f = lay.ICNNLayer(4, None, 7)
        >>> x = torch.randn(10, 4)
        >>> fx, dfdx, d2fd2x = f(x, do_gradient=True, do_Hessian=True)
        >>> print(fx.shape, dfdx.shape, d2fd2x.shape)
        torch.Size([10, 11]) torch.Size([10, 4, 11]) torch.Size([10, 4, 4, 11])

    """

    def __init__(self, input_dim: int, in_features: Union[int, None], out_features: int,
                 act: act.hessQuikActivationFunction = act.softplusActivation(),
                 device=None, dtype=None) -> None:
        r"""

        :param input_dim: dimension of network inputs
        :type input_dim: int
        :param in_features: number of input features. For first ICNN layer, set ``in_features = None``
        :type in_features: int or``None``
        :param out_features: number of output features
        :type out_features: int
        :param act: activation function
        :type act: hessQuikActivationFunction
        :var K: weight matrix for the network inputs of size :math:`(d, n_{out})`
        :var b: bias vector of size :math:`(n_{out},)`
        :var L: weight matrix for the input features of size :math:`(n_{in}, n_{out})`
        :var nonneg: pointwise function to force :math:`l` to have nonnegative weights. Default: ``torch.nn.functional.softplus``
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(ICNNLayer, self).__init__()

        self.input_dim = input_dim
        self.in_features = in_features
        self.out_features = out_features
        self.act = act

        # extract nonnegative weights
        self.nonneg = F.softplus

        self.K = nn.Parameter(torch.empty(input_dim, out_features, **factory_kwargs))

        if in_features is not None:
            self.L = nn.Parameter(torch.empty(in_features, out_features, **factory_kwargs))
        else:
            self.register_parameter('L', None)

        self.b = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.K, a=math.sqrt(self.input_dim))

        if self.L is not None:
            nn.init.kaiming_uniform_(self.L, a=math.sqrt(self.in_features))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.L)
        else:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.K)

        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.b, -bound, bound)

    def dim_input(self) -> int:
        r"""
        number of input features + dimension of network inputs
        """
        n = self.input_dim
        if self.in_features is not None:
            n += self.in_features
        return n

    def dim_output(self) -> int:
        r"""
        number of output features + dimension of network inputs
        """
        return self.out_features + self.input_dim

    def forward(self, ux: torch.Tensor, do_gradient: bool = False, do_Hessian: bool = False, forward_mode: bool = True,
                dudx: Union[torch.Tensor, None] = None, d2ud2x: Union[torch.Tensor, None] = None) \
            -> Tuple[torch.Tensor, Union[torch.Tensor, None], Union[torch.Tensor, None]]:
        r"""
        Forward propagation through ICNN layer of the form

        .. math::

            f(x) =
            \left[\begin{array}{c} \sigma\left(\left[\begin{array}{c}u(x) & x\end{array}\right]
            \left[\begin{array}{c}L^+ \\ K\end{array}\right] + b\right) & x \end{array}\right]

        Here, :math:`u(x)` is the input into the layer of size :math:`(n_s, n_{in})` which is
        a function of the input of the network, :math:`x` of size :math:`(n_s, d)`.
        The output features, :math:`f(x)`, are of size :math:`(n_s, n_{out} + d)`.
        The notation :math:`(\cdot)^+` is a function that makes the weights of a matrix nonnegative.

        As an example, for one sample, :math:`n_s = 1`, the gradient with respect to
        :math:`\begin{bmatrix} u & x \end{bmatrix}` is of the form

        .. math::

            \nabla_x f = \text{diag}\left(\sigma'\left(\left[\begin{array}{c}u(x) & x\end{array}\right]
            \left[\begin{array}{c}L^+ \\ K\end{array}\right] + b\right)\right)
            \left[\begin{array}{c}(L^+)^\top & K^\top\end{array}\right]
            \left[\begin{array}{c}\nabla_x u \\ I\end{array}\right]

        where :math:`\text{diag}` transforms a vector into the entries of a diagonal matrix and :math:`I` is
        the :math:`d \times d` identity matrix.

        """

        (dfdx, d2fd2x) = (None, None)

        M = self.K
        if self.L is not None:
            M = torch.cat((self.nonneg(self.L), M), dim=0)

        z = ux @ M + self.b

        # forward pass
        f, dsig, d2sig = self.act.forward(z, do_gradient=do_gradient, do_Hessian=do_Hessian,
                                          forward_mode=True if forward_mode is True else None)
        f = torch.cat((f, ux[:, -self.input_dim:]), dim=1)

        if (do_gradient or do_Hessian) and forward_mode is True:
            dfdx = dsig.unsqueeze(1) * M

            # -------------------------------------------------------------------------------------------------------- #
            if do_Hessian:
                d2fd2x = (d2sig.unsqueeze(1) * M).unsqueeze(2) * M.unsqueeze(0).unsqueeze(0)

                # Gauss-Newton approximation
                if dudx is not None:
                    d2fd2x = dudx.unsqueeze(1) @ (d2fd2x.permute(0, 3, 1, 2) @ dudx.unsqueeze(1).permute(0, 1, 3, 2))
                    d2fd2x = d2fd2x.permute(0, 2, 3, 1)

                if d2ud2x is not None:
                    # extra term to compute full Hessian
                    d2fd2x += d2ud2x @ dfdx.unsqueeze(1)

                # concatenate zeros
                Z = torch.zeros(d2fd2x.shape[0], d2fd2x.shape[1], d2fd2x.shape[2], self.input_dim,
                                dtype=d2fd2x.dtype, device=d2fd2x.device)
                d2fd2x = torch.cat((d2fd2x, Z), dim=-1)
            # -------------------------------------------------------------------------------------------------------- #

            # finish computing gradient
            if dudx is not None:
                dfdx = dudx @ dfdx

            I = torch.ones(dfdx.shape[0], 1, 1, dtype=dfdx.dtype, device=dfdx.device) \
                * torch.eye(self.input_dim, dtype=dfdx.dtype, device=dfdx.device).unsqueeze(0)
            dfdx = torch.cat((dfdx, I), dim=-1)

        if (do_gradient or do_Hessian) and forward_mode is False:
            dfdx, d2fd2x = self.backward(do_Hessian=do_Hessian)

        return f, dfdx, d2fd2x

    def backward(self, do_Hessian: bool = False,
                 dgdf: Union[torch.Tensor, None] = None, d2gd2f: Union[torch.Tensor, None] = None)\
            -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        r"""
        Backward propagation through ICNN layer of the form

        .. math::

            f(u) =
            \left[\begin{array}{c} \sigma\left(\left[\begin{array}{c}u & x\end{array}\right]
            \left[\begin{array}{c}L^+ \\ K\end{array}\right] + b\right) & x \end{array}\right]

        Here, the network is :math:`g` is a function of :math:`f(u)`.

        As an example, for one sample, :math:`n_s = 1`, the gradient of the network with respect to :math:`u` is of the form

        .. math::

            \nabla_{[u,x]} g = \left(\sigma'\left(\left[\begin{array}{c}u & x\end{array}\right]
            \left[\begin{array}{c}L^+ \\ K\end{array}\right] + b\right) \odot \nabla_{[f, x]} g\right)
            \left[\begin{array}{c}(L^+)^\top & K^\top\end{array}\right]

        where :math:`\odot` denotes the pointwise product.

        """
        M = self.K
        if self.L is not None:
            M = torch.cat((self.nonneg(self.L), M), dim=0)

        # obtain stored information from backward pass
        d2gd2ux = None
        dsig, d2sig = self.act.backward(do_Hessian=do_Hessian)

        # compute gradient
        dgdux = dsig.unsqueeze(1) * M

        # augment gradient
        M2 = torch.ones(dgdux.shape[0], 1, 1, dtype=dgdux.dtype, device=dgdux.device) \
            * torch.eye(self.input_dim, dtype=dgdux.dtype, device=dgdux.device).unsqueeze(0)

        if self.in_features is not None:
            Z = torch.zeros(dgdux.shape[0], self.input_dim, self.in_features)
            M2 = torch.cat((Z, M2), dim=-1).permute(0, 2, 1)

        dgdux = torch.cat((dgdux, M2), dim=-1)

        if do_Hessian:
            # TODO: change order of operations, multiply K's first; check if logic with better naming
            d2gd2ux = (d2sig.unsqueeze(1) * M.unsqueeze(0)).unsqueeze(2) * M.unsqueeze(0).unsqueeze(0)

            # concatenate zeros
            Z = torch.zeros(d2gd2ux.shape[0], d2gd2ux.shape[1], d2gd2ux.shape[2], self.input_dim,
                            dtype=d2gd2ux.dtype, device=d2gd2ux.device)
            d2gd2ux = torch.cat((d2gd2ux, Z), dim=-1)

            if d2gd2f is not None:
                # Gauss-Newton approximation
                h1 = (dgdux.unsqueeze(1) @ d2gd2f.permute(0, 3, 1, 2) @ dgdux.permute(0, 2, 1).unsqueeze(1))
                h1 = h1.permute(0, 2, 3, 1)

                # extra term to compute full Hessian
                N, _, _, m = d2gd2ux.shape
                h2 = d2gd2ux.view(N, -1, m) @ dgdf.view(N, m, -1)
                h2 = h2.view(h1.shape)

                # combine
                d2gd2ux = h1 + h2

        # finish computing gradient
        if dgdf is not None:
            dgdux = dgdux @ dgdf

        return dgdux, d2gd2ux


if __name__ == '__main__':
    from hessQuik.utils import input_derivative_check
    torch.set_default_dtype(torch.float64)

    nex = 11  # no. of examples
    d = 3  # no. of input features
    m = 5  # no. of output features
    x = torch.randn(nex, d)
    f = ICNNLayer(d, None, m, act=act.softplusActivation())

    print('======= FORWARD =======')
    input_derivative_check(f, x, do_Hessian=True, verbose=True, forward_mode=True)

    print('======= BACKWARD =======')
    input_derivative_check(f, x, do_Hessian=True, verbose=True, forward_mode=False)
