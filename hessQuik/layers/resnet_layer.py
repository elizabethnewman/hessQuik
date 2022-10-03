import torch
from hessQuik.layers import hessQuikLayer
import hessQuik.activations as act
from hessQuik.layers import singleLayer
from typing import Union, Tuple


class resnetLayer(hessQuikLayer):
    r"""
    Evaluate and compute derivatives of a residual layer.

    Examples::

        >>> import hessQuik.layers as lay
        >>> f = lay.resnetLayer(4, h=0.25)
        >>> x = torch.randn(10, 4)
        >>> fx, dfdx, d2fd2x = f(x, do_gradient=True, do_Hessian=True)
        >>> print(fx.shape, dfdx.shape, d2fd2x.shape)
        torch.Size([10, 4]) torch.Size([10, 4, 4]) torch.Size([10, 4, 4, 4])

    """

    def __init__(self, width: int, h: float = 1.0, act: act.hessQuikActivationFunction = act.identityActivation(),
                 device=None, dtype=None) -> None:
        r"""
        :param width: number of input and output features, :math:`w`
        :type width: int
        :param h: step size, :math:`h > 0`
        :type h: float
        :param act: activation function
        :type act: hessQuikActivationFunction

        :var layer: singleLayer with :math:`w` input features and :math:`w` output features
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(resnetLayer, self).__init__()

        self.width = width
        self.h = h
        self.layer = singleLayer(width, width, act=act, **factory_kwargs)

    def dim_input(self) -> int:
        r"""
        width
        """
        return self.width

    def dim_output(self) -> int:
        r"""
        width
        """
        return self.width

    def forward(self, u, do_gradient=False, do_Hessian=False, do_Laplacian=False, forward_mode=True,
                dudx=None, d2ud2x=None, v=None):
        r"""
        Forward propagation through resnet layer of the form

        .. math::

            f(x) = u(x) + h \cdot singleLayer(u(x))

        Here, :math:`u(x)` is the input into the layer of size :math:`(n_s, w)` which is
        a function of the input of the network, :math:`x`.
        The output features, :math:`f(x)`, are of size :math:`(n_s, w)`.

        As an example, for one sample, :math:`n_s = 1`, the gradient with respect to :math:`x` is of the form

<<<<<<< HEAD
    def forward(self, u, do_gradient=False, do_Hessian=False, do_Laplacian=False, dudx=None, d2ud2x=None, lap_u=None):

        (dfdx, d2fd2x, lap_f) = (None, None, None)
        fi, dfi, d2fi, lap_fi = self.layer(u, do_gradient=do_gradient, do_Hessian=do_Hessian, do_Laplacian=do_Laplacian,
                                           dudx=dudx, d2ud2x=d2ud2x, lap_u=lap_u)
=======
        .. math::

            \nabla_x f = I + h \nabla_x singleLayer(u(x))

        where :math:`I` denotes the :math:`w \times w` identity matrix.
        """

        if do_Laplacian and not do_Hessian:
            forward_mode = True

        (dfdx, d2fd2x) = (None, None)
<<<<<<< HEAD
        fi, dfi, d2fi = self.layer(u, do_gradient=do_gradient, do_Hessian=do_Hessian, dudx=dudx, d2ud2x=d2ud2x,
                                   forward_mode=True if forward_mode is True else None)
>>>>>>> c846faf2d50607569f3f073aa019d49e967371c4
=======
        fi, dfi, d2fi = self.layer(u, do_gradient=do_gradient, do_Hessian=do_Hessian, do_Laplacian=do_Laplacian,
                                   dudx=dudx, d2ud2x=d2ud2x,
                                   forward_mode=True if forward_mode is True else None, v=v)
>>>>>>> main

        # skip connection
        f = u + self.h * fi

        if do_gradient and forward_mode is True:

            if dudx is None:
                if v is None:
                    v = torch.eye(self.width, dtype=dfi.dtype, device=dfi.device)
                dfdx = v + self.h * dfi
            else:
                dfdx = dudx + self.h * dfi

        if (do_Hessian or do_Laplacian) and forward_mode is True:
            d2fd2x = self.h * d2fi
            if d2ud2x is not None:
                d2fd2x += d2ud2x

<<<<<<< HEAD
        if (do_gradient or do_Hessian) and self.reverse_mode is True:
            dfdx, d2fd2x, lap_f = self.backward(do_Hessian=do_Hessian, do_Laplacian=do_Laplacian)
=======
        if (do_gradient or do_Hessian) and forward_mode is False:
<<<<<<< HEAD
            dfdx, d2fd2x = self.backward(do_Hessian=do_Hessian)
>>>>>>> c846faf2d50607569f3f073aa019d49e967371c4
=======
            dfdx, d2fd2x = self.backward(do_Hessian=do_Hessian, v=v)
>>>>>>> main

        return f, dfdx, d2fd2x, lap_f

<<<<<<< HEAD
<<<<<<< HEAD
    def backward(self, do_Hessian=False, do_Laplacian=False, dgdf=None, d2gd2f=None, lap_g=None):
        d2gd2x = None
        if not do_Hessian:

            dgdx = self.layer.backward(do_Hessian=False, do_Laplacian=do_Laplacian,
                                       dgdf=dgdf, d2gd2f=None, lap_g=None)[0]
=======
    def backward(self, do_Hessian: bool = False,
                 dgdf: Union[torch.Tensor, None] = None, d2gd2f: Union[torch.Tensor, None] = None):
=======
    def backward(self, do_Hessian=False, dgdf=None, d2gd2f=None, v=None):
>>>>>>> main
        r"""
        Backward propagation through single layer of the form

        .. math::

                f(u) = u + h \cdot singleLayer(u)

        Here, the network is :math:`g` is a function of :math:`f(u)`.

        As an example, for one sample, :math:`n_s = 1`, the gradient of the network with respect to :math:`u` is of the form

        .. math::

            \nabla_u g = \nabla_f g + h \cdot \nabla_u singleLayer(u)

        where :math:`\odot` denotes the pointwise product.

        """
        
        d2gd2u = None
        if not do_Hessian:

<<<<<<< HEAD
            dgdu = self.layer.backward(do_Hessian=False, dgdf=dgdf, d2gd2f=None)[0]
>>>>>>> c846faf2d50607569f3f073aa019d49e967371c4
=======
            dgdu = self.layer.backward(do_Hessian=False, dgdf=dgdf, d2gd2f=None, v=v)[0]
>>>>>>> main

            if dgdf is None:
                if v is None:
                    v = torch.eye(self.width, dtype=dgdu.dtype, device=dgdu.device)

                dgdu = v + self.h * dgdu
            else:
                dgdu = dgdf + self.h * dgdu
        else:
<<<<<<< HEAD
            dfdx, d2fd2x = self.layer.backward(do_Hessian=do_Hessian, do_Laplacian=do_Laplacian,
                                               dgdf=None, d2gd2f=None, lap_g=None)[:2]
=======
            dfdx, d2fd2x = self.layer.backward(do_Hessian=do_Hessian, dgdf=None, d2gd2f=None, v=v)[:2]
>>>>>>> main

            if v is None:
                v = torch.eye(self.width, dtype=dfdx.dtype, device=dfdx.device)

            dgdu = v + self.h * dfdx
            if dgdf is not None:
                dgdu = dgdu @ dgdf

            # d2gd2u = self.h * d2fd2x
            if d2gd2f is None:
                d2gd2u = self.h * d2fd2x
            else:
                # TODO: compare timings for h_dfdx on CPU and GPU
                h_dfdx = torch.eye(self.width, dtype=dfdx.dtype, device=dfdx.device) + self.h * dfdx

                # Gauss-Newton approximation
                h1 = (h_dfdx.unsqueeze(1) @ d2gd2f.permute(0, 3, 1, 2) @ h_dfdx.permute(0, 2, 1).unsqueeze(1))
                h1 = h1.permute(0, 2, 3, 1)

                # extra term to compute full Hessian
                h2 = d2fd2x @ dgdf.unsqueeze(1)
                # combine
                d2gd2u = h1 + self.h * h2

<<<<<<< HEAD
        return dgdx, d2gd2x, lap_g
=======
        return dgdu, d2gd2u
>>>>>>> c846faf2d50607569f3f073aa019d49e967371c4

    def extra_repr(self) -> str:
        r"""
        :meta private:
        """
        return 'width={}, h={}'.format(self.width, self.h)


if __name__ == '__main__':
    from hessQuik.utils import input_derivative_check, directional_derivative_check, directional_derivative_laplacian_check, input_derivative_check_finite_difference_laplacian
    torch.set_default_dtype(torch.float64)

    nex = 11  # no. of examples
    width = 4  # no. of input features
    h = 0.25
    x = torch.randn(nex, width)
    f = resnetLayer(width, h=h, act=act.softplusActivation())

    print('\n======= FORWARD =======')
    input_derivative_check(f, x, do_Hessian=True, verbose=True, forward_mode=True)

    print('\n======= BACKWARD =======')
    input_derivative_check(f, x, do_Hessian=True, verbose=True, forward_mode=False)

    print('\n======= LAPLACIAN =======')
    input_derivative_check_finite_difference_laplacian(f, x, do_Laplacian=True, verbose=True)

    print('\n======= DIRECTIONAL =======')
    directional_derivative_check(f, x, verbose=True)
    directional_derivative_laplacian_check(f, x, verbose=True)