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
                 device=None, dtype=None) -> None:
        r"""
        :param in_features: number of input features, :math:`n_{in}`
        :type in_features: int
        :param out_features: number of output features, :math:`n_{out}`
        :type out_features: int
        :param act: activation function
        :type act: hessQuikActivationFunction
        :var K: weight matrix of size :math:`(n_{in}, n_{out})`
        :var b: bias vector of size :math:`(n_{out},)`
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(singleLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.act = act

        self.K = nn.Parameter(torch.empty(in_features, out_features, **factory_kwargs))
        self.b = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.K, a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.K)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
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

    def forward(self, u: torch.Tensor, do_gradient: bool = False, do_Hessian: bool = False, forward_mode: bool = True,
                dudx: Union[torch.Tensor, None] = None, d2ud2x: Union[torch.Tensor, None] = None) \
            -> Tuple[torch.Tensor, Union[torch.Tensor, None], Union[torch.Tensor, None]]:
        r"""
        Forward propagation through single layer of the form

        .. math::

<<<<<<< HEAD
    def forward(self, u, do_gradient=False, do_Hessian=False, do_Laplacian=False, dudx=None, d2ud2x=None, lap_u=None):

        (dfdx, d2fd2x, lap_f) = (None, None, None)
        f, dsig, d2sig = self.act.forward(u @ self.K + self.b,
                                          do_gradient=do_gradient, do_Hessian=do_Hessian, do_Laplacian=do_Laplacian)
=======
            f(x) = \sigma(u(x) K + b)

        Here, :math:`u(x)` is the input into the layer of size :math:`(n_s, n_{in})` which is
        a function of the input of the network, :math:`x`.
        The output features, :math:`f(x)`, are of size :math:`(n_s, n_{out})`.

        As an example, for one sample, :math:`n_s = 1`, the gradient with respect to :math:`x` is of the form

        .. math::

            \nabla_x f = \text{diag}(\sigma'(u(x) K + b))K^\top \nabla_x u
>>>>>>> c846faf2d50607569f3f073aa019d49e967371c4

        where :math:`\text{diag}` transforms a vector into the entries of a diagonal matrix.
        """
        (dfdx, d2fd2x) = (None, None)
        f, dsig, d2sig = self.act(u @ self.K + self.b, do_gradient=do_gradient, do_Hessian=do_Hessian,
                                  forward_mode=True if forward_mode is True else None)
        # ------------------------------------------------------------------------------------------------------------ #
        # forward mode
<<<<<<< HEAD
        if (do_gradient or do_Hessian or do_Laplacian) and self.reverse_mode is False:
=======
        if (do_gradient or do_Hessian) and forward_mode is True:
>>>>>>> c846faf2d50607569f3f073aa019d49e967371c4
            dfdx = dsig.unsqueeze(1) * self.K
            # -------------------------------------------------------------------------------------------------------- #
            if do_Hessian:
                d2fd2x = (d2sig.unsqueeze(1) * self.K).unsqueeze(2) * self.K.unsqueeze(0).unsqueeze(0)

                # Gauss-Newton approximation
                if d2ud2x is not None:
                    d2fd2x = dudx.unsqueeze(1) @ (d2fd2x.permute(0, 3, 1, 2) @ dudx.permute(0, 2, 1).unsqueeze(1))
                    d2fd2x = d2fd2x.permute(0, 2, 3, 1)

                    # extra term to compute full Hessian
<<<<<<< HEAD
                    d2fd2x += d2ud2x @ dfdx.unsqueeze(1)  # I already compute this in gradient

                if do_Laplacian:
                    lap_f = d2fd2x[:, torch.arange(d2fd2x.shape[1]), torch.arange(d2fd2x.shape[1]), :].sum(1)

            if do_Laplacian and not do_Hessian:

                if dudx is None:
                    lap_f = ((self.K ** 2).unsqueeze(0) * d2sig.unsqueeze(1)).sum(1)

                if lap_u is not None:
                    # lap_u = d2ud2x[:, torch.arange(d2ud2x.shape[1]), torch.arange(d2ud2x.shape[1]), :].sum(1)
                    lap1 = (lap_u.unsqueeze(-1) * dfdx).sum(1)
                    lap2 = (((dudx @ self.K) ** 2) * d2sig.unsqueeze(1)).sum(1)  # TODO: can we speed this up?
                    lap_f = lap1 + lap2
=======
                    d2fd2x += d2ud2x @ dfdx.unsqueeze(1)
>>>>>>> c846faf2d50607569f3f073aa019d49e967371c4
            # -------------------------------------------------------------------------------------------------------- #
            # finish computing gradient
            if dudx is not None:
                dfdx = dudx @ dfdx

        # ------------------------------------------------------------------------------------------------------------ #
        # backward mode (if layer is not wrapped in NN)
<<<<<<< HEAD
        if (do_gradient or do_Hessian or do_Laplacian) and self.reverse_mode is True:
            dfdx, d2fd2x, lap_f = self.backward(do_Hessian=do_Hessian, do_Laplacian=do_Laplacian)
=======
        if (do_gradient or do_Hessian) and forward_mode is False:
            dfdx, d2fd2x = self.backward(do_Hessian=do_Hessian)
>>>>>>> c846faf2d50607569f3f073aa019d49e967371c4

        return f, dfdx, d2fd2x, lap_f

<<<<<<< HEAD
    def backward(self, do_Hessian=False, do_Laplacian=False, dgdf=None, d2gd2f=None, lap_g=None):
        (d2gd2x, lap_gx) = (None, None)
        dsig, d2sig = self.act.backward(do_Hessian=do_Hessian, do_Laplacian=do_Laplacian)
        dgdx = dsig.unsqueeze(1) * self.K
=======
    def backward(self, do_Hessian: bool = False,
                 dgdf: Union[torch.Tensor, None] = None, d2gd2f: Union[torch.Tensor, None] = None):
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
        dgdu = dsig.unsqueeze(1) * self.K
>>>>>>> c846faf2d50607569f3f073aa019d49e967371c4

        if do_Hessian:
            d2gd2u = (d2sig.unsqueeze(1) * self.K.unsqueeze(0)).unsqueeze(2) * self.K.unsqueeze(0).unsqueeze(0)

            if d2gd2f is not None:
                # Gauss-Newton approximation
                h1 = (dgdu.unsqueeze(1) @ d2gd2f.permute(0, 3, 1, 2) @ dgdu.permute(0, 2, 1).unsqueeze(1))
                h1 = h1.permute(0, 2, 3, 1)

                # extra term to compute full Hessian
                h2 = d2gd2u @ dgdf.unsqueeze(1)
                # combine
                d2gd2u = h1 + h2

                if do_Laplacian:
                    lap_gx = d2gd2x[:, torch.arange(d2gd2x.shape[1]), torch.arange(d2gd2x.shape[1]), :].sum(1)

        if do_Laplacian and not do_Hessian:

            if dgdf is None:
                lap_gx = ((self.K ** 2).unsqueeze(0) * d2sig.unsqueeze(1)).sum(1)

            if lap_g is not None:
                # lap_u = d2ud2x[:, torch.arange(d2ud2x.shape[1]), torch.arange(d2ud2x.shape[1]), :].sum(1)
                lap1 = (lap_g.unsqueeze(-1) * dgdx).sum(1)
                lap2 = (((dgdx @ self.K) ** 2) * d2sig.unsqueeze(1)).sum(1)  # TODO: can we speed this up?
                lap_gx = lap1 + lap2

        # finish computing gradient
        if dgdf is not None:
            dgdu = dgdu @ dgdf

<<<<<<< HEAD
        return dgdx, d2gd2x, lap_gx
=======
        return dgdu, d2gd2u
>>>>>>> c846faf2d50607569f3f073aa019d49e967371c4

    def extra_repr(self) -> str:
        r"""
        :meta private:
        """
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )


if __name__ == '__main__':
    from hessQuik.utils import input_derivative_check, laplacian_check_using_hessian
    torch.set_default_dtype(torch.float64)

    nex = 11  # no. of examples
    d = 4  # no. of input features
    m = 7  # no. of output features
    x = torch.randn(nex, d)

    f = singleLayer(d, m, act=act.softplusActivation())

    print('======= FORWARD =======')
    input_derivative_check(f, x, do_Hessian=True, verbose=True, forward_mode=True)

    print('======= BACKWARD =======')
<<<<<<< HEAD
    f = singleLayer(d, m, act=act.softplusActivation())
    f.reverse_mode = True
    input_derivative_check(f, x, do_Hessian=True, verbose=True)

    print('======= LAPLACIAN: FORWARD =======')
    f.reverse_mode = False
    laplacian_check_using_hessian(f, x)

    print('======= LAPLACIAN: BACKWARD =======')
    f.reverse_mode = True
    laplacian_check_using_hessian(f, x)
=======
    input_derivative_check(f, x, do_Hessian=True, verbose=True, forward_mode=False)
>>>>>>> c846faf2d50607569f3f073aa019d49e967371c4
