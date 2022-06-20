import torch
from hessQuik.layers import lapQuikLayer
import hessQuik.activations as act
from hessQuik.layers import singleLayerLapQuik
from typing import Union, Tuple


class resnetLayerLapQuik(lapQuikLayer):
    r"""
    Evaluate and compute derivatives of a residual layer.

    Examples::

        >>> import hessQuik.layers as lay
        >>> f = lay.resnetLayerLapQuik(4, h=0.25)
        >>> x = torch.randn(10, 4)
        >>> fx, dfdx, lapfd2x = f(x, do_gradient=True, do_Laplacian=True)
        >>> print(fx.shape, dfdx.shape, lapfd2x.shape)
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
        :type act: lapQuikActivationFunction

        :var layer: singleLayer with :math:`w` input features and :math:`w` output features
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(resnetLayerLapQuik, self).__init__()

        self.width = width
        self.h = h
        self.layer = singleLayerLapQuik(width, width, act=act, **factory_kwargs)

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

    def forward(self, u: torch.Tensor, do_gradient: bool = False, do_Laplacian: bool = False,
                dudx: Union[torch.Tensor, None] = None, lapud2x: Union[torch.Tensor, None] = None) \
            -> Tuple[torch.Tensor, Union[torch.Tensor, None], Union[torch.Tensor, None]]:
        r"""
        Forward propagation through resnet layer of the form

        .. math::

            f(x) = u(x) + h \cdot singleLayer(u(x))

        Here, :math:`u(x)` is the input into the layer of size :math:`(n_s, w)` which is
        a function of the input of the network, :math:`x`.
        The output features, :math:`f(x)`, are of size :math:`(n_s, w)`.

        As an example, for one sample, :math:`n_s = 1`, the gradient with respect to :math:`x` is of the form

        .. math::

            \nabla_x f = I + h \nabla_x singleLayer(u(x))

        where :math:`I` denotes the :math:`w \times w` identity matrix.
        """
        (dfdx, lapfd2x) = (None, None)
        fi, dfi, lapfi = self.layer(u, do_gradient=do_gradient, do_Laplacian=do_Laplacian, dudx=dudx, lapud2x=lapud2x)

        # skip connection
        f = u + self.h * fi

        if do_gradient:

            if dudx is None:
                dfdx = torch.eye(self.width, dtype=dfi.dtype, device=dfi.device) + self.h * dfi
            else:
                dfdx = dudx + self.h * dfi

        if do_Laplacian:
            lapfd2x = self.h * lapfi
            if lapud2x is not None:
                lapfd2x += lapud2x

        return f, dfdx, lapfd2x

    def extra_repr(self) -> str:
        r"""
        :meta private:
        """
        return 'width={}, h={}'.format(self.width, self.h)


if __name__ == '__main__':
    from hessQuik.utils import input_derivative_check_finite_difference_laplacian
    torch.set_default_dtype(torch.float64)

    nex = 11  # no. of examples
    width = 4  # no. of input features
    h = 0.25
    x = torch.randn(nex, width)
    f = resnetLayerLapQuik(width, h=h, act=act.softplusActivation())

    input_derivative_check_finite_difference_laplacian(f, x, do_Laplacian=True, verbose=True)
