import torch
import torch.nn as nn
import math
from hessQuik.layers import lapQuikLayer
import hessQuik.activations as act
from typing import Union, Tuple
from torch import Tensor


class singleLayerLapQuik(lapQuikLayer):
    r"""
    Evaluate and compute derivatives of a single layer.

    Examples::

        >>> import hessQuik.layers as lay
        >>> f = lay.singleLayerLapQuik(4, 7)
        >>> x = torch.randn(10, 4)
        >>> fx, dfdx, lapfd2x = f(x, do_gradient=True, do_Laplacian=True)
        >>> print(fx.shape, dfdx.shape, lapfd2x.shape)
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
        super(singleLayerLapQuik, self).__init__()

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

    def forward(self, u: Tensor, do_gradient: bool = False, do_Laplacian: bool = False,
                dudx: Union[Tensor, None] = None, lapud2x: Union[Tensor, None] = None) -> Tuple[Tensor, Union[Tensor, None], Union[Tensor, None]]:
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
        (dfdx, lapfd2x) = (None, None)
        f, dsig, d2sig = self.act(u @ self.K + self.b, do_gradient=do_gradient, do_Hessian=do_Laplacian,
                                  forward_mode=True)
        # ------------------------------------------------------------------------------------------------------------ #
        # forward mode
        if do_gradient or do_Laplacian:
            dfdx = dsig.unsqueeze(1) * self.K
            # -------------------------------------------------------------------------------------------------------- #
            if do_Laplacian:
                # d2fd2x = (d2sig.unsqueeze(1) * self.K).unsqueeze(2) * self.K.unsqueeze(0).unsqueeze(0)
                if lapud2x is None:
                    lapfd2x = (d2sig.unsqueeze(1) * (torch.sum(self.K ** 2, dim=0, keepdim=True)))
                else:
                    lapfd2x = (d2sig.unsqueeze(1) * (torch.sum((dudx @ self.K) ** 2, dim=1, keepdim=True)))
                    # extra term to compute full Hessian
                    lapfd2x += (lapud2x.unsqueeze(1) @ dfdx.unsqueeze(1)).squeeze(1)
            # -------------------------------------------------------------------------------------------------------- #
            # finish computing gradient
            if dudx is not None:
                dfdx = dudx @ dfdx

        return f, dfdx, lapfd2x

    def extra_repr(self) -> str:
        r"""
        :meta private:
        """
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )


if __name__ == '__main__':
    from hessQuik.utils import input_derivative_check_finite_difference_laplacian
    torch.set_default_dtype(torch.float64)

    nex = 11  # no. of examples
    d = 4  # no. of input features
    m = 7  # no. of output features
    x = torch.randn(nex, d)

    f = singleLayerLapQuik(d, m, act=act.softplusActivation())

    input_derivative_check_finite_difference_laplacian(f, x, do_Laplacian=True, verbose=True)
