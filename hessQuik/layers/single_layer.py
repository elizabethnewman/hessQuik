import torch
import torch.nn as nn
import math
from hessQuik.layers import hessQuikLayer
import hessQuik.activations as act
from typing import Union, Tuple


class singleLayer(hessQuikLayer):
    r"""
    Evaluate and compute derivatives of a single layer.

    Examples::

        >>> import hessQuik.layers as lay
        >>> f = lay.singleLayer(4, 7)
        >>> x = torch.randn(10, 4)
        >>> fx, dfdx, d2fd2x = f(x, do_gradient=True, do_Hessian=True)
        >>> print(fx.shape, dfdx.shape, d2fd2x.shape)
        torch.Size([10, 7]) torch.Size([10, 4, 7]) torch.Size([10, 4, 4, 7])

    """

    def __init__(self, in_features: int, out_features: int,
                 act: act.hessQuikActivationFunction = act.identityActivation(),
                 bias: bool = True,
                 device=None, dtype=None) -> None:
        r"""
        :param in_features: number of input features, :math:`n_{in}`
        :type in_features: int
        :param out_features: number of output features, :math:`n_{out}`
        :type out_features: int
        :param act: activation function
        :type act: hessQuikActivationFunction
        :param bias: additive bias
        :type bias: bool
        :var K: weight matrix of size :math:`(n_{in}, n_{out})`
        :var b: bias vector of size :math:`(n_{out},)`
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(singleLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.act = act
        self.bias = bias

        self.K = nn.Parameter(torch.empty(in_features, out_features, **factory_kwargs))

        if self.bias:
            self.b = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('b', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.K, a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.K)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0

        if self.bias:
            nn.init.uniform_(self.b, -bound, bound)

    def dim_input(self) -> int:
        r"""
        number of input features
        """
        return self.in_features

    def dim_output(self):
        r"""
        number of output features
        """
        return self.out_features

    def forward(self, u, do_gradient=False, do_Hessian=False, do_Laplacian=False, forward_mode=True,
                dudx=None, d2ud2x=None, v=None):
        r"""
        Forward propagation through single layer of the form

        .. math::

            f(x) = \sigma(u(x) K + b)

        Here, :math:`u(x)` is the input into the layer of size :math:`(n_s, n_{in})` which is
        a function of the input of the network, :math:`x`.
        The output features, :math:`f(x)`, are of size :math:`(n_s, n_{out})`.

        As an example, for one sample, :math:`n_s = 1`, the gradient with respect to :math:`x` is of the form

        .. math::

            \nabla_x f = \text{diag}(\sigma'(u(x) K + b))K^\top \nabla_x u

        where :math:`\text{diag}` transforms a vector into the entries of a diagonal matrix.
        """
        if do_Laplacian and not do_Hessian:
            forward_mode = True

        (dfdx, d2fd2x) = (None, None)

        z = u @ self.K
        if self.bias:
            z += self.b

        f, dsig, d2sig = self.act(z, do_gradient=do_gradient, do_Hessian=do_Hessian or do_Laplacian,
                                  forward_mode=True if forward_mode is True else None)
        # ------------------------------------------------------------------------------------------------------------ #
        # forward mode
        if (do_gradient or do_Hessian or do_Laplacian) and forward_mode is True:
            K = self.K

            # apply direction
            if v is not None:
                K = v @ self.K

            dfdx = dsig.unsqueeze(1) * K
            # -------------------------------------------------------------------------------------------------------- #
            if do_Laplacian and not do_Hessian:
                if d2ud2x is None:
                    d2fd2x = (d2sig.unsqueeze(1) * (torch.sum(K ** 2, dim=0, keepdim=True)))
                else:
                    d2fd2x = (d2sig.unsqueeze(1) * (torch.sum((dudx @ K) ** 2, dim=1, keepdim=True)))
                    d2fd2x += (d2ud2x.unsqueeze(1) @ dfdx.unsqueeze(1)).squeeze(1)
            # -------------------------------------------------------------------------------------------------------- #
            if do_Hessian:
                d2fd2x = (d2sig.unsqueeze(1) * K).unsqueeze(2) * K.unsqueeze(0).unsqueeze(0)

                # Gauss-Newton approximation
                if d2ud2x is not None:
                    d2fd2x = dudx.unsqueeze(1) @ (d2fd2x.permute(0, 3, 1, 2) @ dudx.permute(0, 2, 1).unsqueeze(1))
                    d2fd2x = d2fd2x.permute(0, 2, 3, 1)

                    # extra term to compute full Hessian
                    d2fd2x += d2ud2x @ dfdx.unsqueeze(1)
            # -------------------------------------------------------------------------------------------------------- #
            # finish computing gradient
            if dudx is not None:
                dfdx = dudx @ dfdx

        # ------------------------------------------------------------------------------------------------------------ #
        # backward mode (if layer is not wrapped in NN)
        if (do_gradient or do_Hessian) and forward_mode is False:
            dfdx, d2fd2x = self.backward(do_Hessian=do_Hessian, v=v)

        return f, dfdx, d2fd2x

    def backward(self, do_Hessian=False, dgdf=None, d2gd2f=None, v=None):
        r"""
        Backward propagation through single layer of the form

        .. math::

                f(u) = \sigma(u K + b)

        Here, the network is :math:`g` is a function of :math:`f(u)`.

        As an example, for one sample, :math:`n_s = 1`, the gradient of the network with respect to :math:`u` is of the form

        .. math::

            \nabla_u g = (\sigma'(u K + b) \odot \nabla_f g)K^\top

        where :math:`\odot` denotes the pointwise product.

        """

        d2gd2u = None
        dsig, d2sig = self.act.backward(do_Hessian=do_Hessian)

        # if v is not None:
        #     dsig = dsig.unsqueeze(2) * v
        #     dgdu = self.K @ dsig
        # else:
        #     # compute gradient
        dgdu = dsig.unsqueeze(1) * self.K

        if v is not None:
            dgdu = dgdu @ v

        if do_Hessian:

            d2gd2u = (d2sig.unsqueeze(1) * self.K.unsqueeze(0)).unsqueeze(2) * self.K.unsqueeze(0).unsqueeze(0)

            if v is not None:
                d2gd2u = d2gd2u @ v

            if d2gd2f is not None:
                # Gauss-Newton approximation
                h1 = (dgdu.unsqueeze(1) @ d2gd2f.permute(0, 3, 1, 2) @ dgdu.permute(0, 2, 1).unsqueeze(1))
                h1 = h1.permute(0, 2, 3, 1)

                # extra term to compute full Hessian
                h2 = d2gd2u @ dgdf.unsqueeze(1)

                # combine
                d2gd2u = h1 + h2

        # finish computing gradient
        if dgdf is not None:
            dgdu = dgdu @ dgdf

        return dgdu, d2gd2u

    def extra_repr(self) -> str:
        r"""
        :meta private:
        """
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )


if __name__ == '__main__':
    from hessQuik.utils import input_derivative_check, directional_derivative_check, directional_derivative_laplacian_check, input_derivative_check_finite_difference_laplacian
    torch.set_default_dtype(torch.float64)

    nex = 11  # no. of examples
    d = 4  # no. of input features
    m = 7  # no. of output features
    x = torch.randn(nex, d)

    f = singleLayer(d, m, act=act.softplusActivation())

    # test directional derivative
    directional_derivative_check(f, x, verbose=True)
    directional_derivative_laplacian_check(f, x, verbose=True)

    # test Laplacian
    input_derivative_check_finite_difference_laplacian(f, x, do_Laplacian=True, verbose=True)

    # print('======= FORWARD =======')
    # input_derivative_check(f, x, do_Hessian=True, verbose=True, forward_mode=True)
    #
    # print('======= BACKWARD =======')
    # input_derivative_check(f, x, do_Hessian=True, verbose=True, forward_mode=False)
