import torch
import torch.nn as nn
from torch import Tensor
from typing import Union, Tuple
import math
import hessQuik.activation_functions as act


class hessQuikLayer(nn.Module):

    def __init__(self, *args, **kwargs):
        super(hessQuikLayer, self).__init__()

    def forward(self, u: Tensor, do_gradient: bool = False, do_Hessian: bool = False,
                dudx: Union[Tensor, None] = None, d2ud2x: Union[Tensor, None] = None,
                reverse_mode: bool = False) \
            -> Tuple[Tensor, Union[Tensor, None], Union[Tensor, None]]:
        """
        Forward propagate through singleLayer

        Parameters
        ----------
        u : (N, in_features) torch.Tensor
            Tensor of input features
        do_gradient : bool, optional
            If True, compute the gradient of the output of the layer, f(u(x)), with respect to the network input, x
        do_Hessian : bool, optional
            If True, compute the Hessian of the output of the layer, f(u(x)), with respect to the network input, x
        dudx : (N, d, in_features) torch.Tensor
            Gradient of the input into the layer, u(x), with respect to the network inputs, x
        d2ud2x : (N, d, d, in_features) torch.Tensor
            Hessian of the input into the layer, u(x), with respect to the network inputs, x

        Returns
        -------
        f : (N, out_features) torch.Tensor
            Output of the layer, f(u(x))
        dfdx : (N, d, out_features)
            Gradient of the output of the layer, f(u(x)), with respect to the network inputs, x
        d2fd2x : (N, d, d, out_features)
            Hessian of the output of the layer, f(u(x)), with respect to the network inputs, x
        """
        raise NotImplementedError

    def backward(self, do_Hessian: bool = False,
                 dgdf: Union[Tensor, None] = None, d2gd2f: Union[Tensor, None] = None) \
            -> Tuple[Tensor, Union[Tensor, None]]:
        """
        Backward propagate through singleLayer

        Parameters
        ----------
        do_Hessian : bool, optional
            If True, compute the Hessian of the output of the layer, g(f(x)), with respect to the network input, x
        dgdf : (N, d, out_features) torch.Tensor
            Gradient of the input into the layer, g(f(x)), with respect to the layer outputs, f(x)
        d2gd2f : (N, d, d, out_features) torch.Tensor
            Hessian of the input into the layer, g(f(x)), with respect to the layer outputs, f(x)

        Returns
        -------
        dgdx : (N, d, in_features)
            Gradient of the output of the layer, g(f(x)), with respect to the network inputs, x
        d2gd2x : (N, d, d, in_features)
            Hessian of the output of the layer, g(f(x)), with respect to the network inputs, x
        """
        raise NotImplementedError


class singleLayer(hessQuikLayer):
    """
    Forward propagation through single layer of the form

        f(u(x)) = act(u(x) @ K + b).

    Here, u(x) is the input into the layer and x is the input into the network of shapes

        x : (N, d) torch.Tensor
        u(x) : (N, in_features) torch.Tensor
        f(u(x)) : (N, out_features) torch.Tensor

    where N is the number of examples and d is the number of input features into the network.
    """

    def __init__(self, in_features, out_features, act: act.activationFunction = act.identityActivation(),
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(singleLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.act = act

        self.K = nn.Parameter(torch.empty(in_features, out_features, **factory_kwargs))
        self.b = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.K, a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.K)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.b, -bound, bound)

    def forward(self, u, do_gradient=False, do_Hessian=False, dudx=None, d2ud2x=None, reverse_mode=False):

        (dfdx, d2fd2x) = (None, None)
        f, dsig, d2sig = self.act.forward(u @ self.K + self.b, do_gradient=do_gradient, do_Hessian=do_Hessian,
                                          reverse_mode=reverse_mode)

        if (do_gradient or do_Hessian) and not reverse_mode:
            dfdx = dsig.unsqueeze(1) * self.K

            # -------------------------------------------------------------------------------------------------------- #
            if do_Hessian:
                d2fd2x = (d2sig.unsqueeze(1) * self.K).unsqueeze(2) * self.K.unsqueeze(0).unsqueeze(0)

                # Gauss-Newton approximation
                if dudx is not None:
                    d2fd2x = dudx.unsqueeze(1) @ (d2fd2x.permute(0, 3, 1, 2) @ dudx.unsqueeze(1).permute(0, 1, 3, 2))
                    d2fd2x = d2fd2x.permute(0, 2, 3, 1)

                if d2ud2x is not None:
                    # extra term to compute full Hessian
                    d2fd2x += d2ud2x @ dfdx.unsqueeze(1)  # I already compute this in gradient
            # -------------------------------------------------------------------------------------------------------- #
            # finish computing gradient
            if dudx is not None:
                dfdx = dudx @ dfdx

        return f, dfdx, d2fd2x

    def backward(self, do_Hessian=False, dgdf=None, d2gd2f=None):

        d2gd2x = None
        dsig, d2sig = self.act.backward(do_Hessian=do_Hessian)
        dgdx = dsig.unsqueeze(1) * self.K

        if do_Hessian:
            # TODO: change order of operations, multiply K's first; check if logic with better naming
            d2gd2x = (d2sig.unsqueeze(1) * self.K.unsqueeze(0)).unsqueeze(2) * self.K.unsqueeze(0).unsqueeze(0)
            if d2gd2f is not None:
                # Gauss-Newton approximation
                h1 = (dgdx.unsqueeze(1) @ d2gd2f.permute(0, 3, 1, 2) @ dgdx.permute(0, 2, 1).unsqueeze(1))
                h1 = h1.permute(0, 2, 3, 1)

                # extra term to compute full Hessian
                N, _, _, m = d2gd2x.shape
                h2 = d2gd2x.view(N, -1, m) @ dgdf.view(N, m, -1)
                h2 = h2.view(h1.shape)

                # combine
                d2gd2x = h1 + h2

        # finish computing gradient
        if dgdf is not None:
            dgdx = dgdx @ dgdf

        return dgdx, d2gd2x

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )


class resnetLayer(hessQuikLayer):
    """
    Forward propagation through residual layer of the form

        f(u(x)) = u(x) + h * layer(u(x)) where layer(u(x)) = act(u(x) @ K + b).

    Here, u(x) is the input into the layer and x is the input into the network of shapes

        x : (N, d) torch.Tensor
        u(x) : (N, width) torch.Tensor
        f(u(x)) : (N, width) torch.Tensor

    where N is the number of examples and d is the number of input features into the network.
    """

    def __init__(self, width, h=1.0, act: activationFunction = identityActivation(), device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(resnetLayer, self).__init__()

        self.width = width
        self.h = h
        self.layer = singleLayer(width, width, act=act, **factory_kwargs)

    def forward(self, u, do_gradient=False, do_Hessian=False, dudx=None, d2ud2x=None, reverse_mode=False):

        (dfdx, d2fd2x) = (None, None)
        fi, dfi, d2fi = self.layer(u, do_gradient=do_gradient, do_Hessian=do_Hessian, dudx=dudx, d2ud2x=d2ud2x,
                                   reverse_mode=reverse_mode)

        # skip connection
        f = u + self.h * fi

        if do_gradient and not reverse_mode:

            if dudx is None:
                dfdx = torch.eye(self.width, dtype=dfi.dtype, device=dfi.device) + self.h * dfi
            else:
                dfdx = dudx + self.h * dfi

        if do_Hessian and not reverse_mode:
            d2fd2x = self.h * d2fi
            if d2ud2x is not None:
                d2fd2x += d2ud2x

        return f, dfdx, d2fd2x

    def backward(self, do_Hessian=False, dgdf=None, d2gd2f=None):
        d2gd2x = None
        if not do_Hessian:

            dgdx = self.layer.backward(do_Hessian=False, dgdf=dgdf, d2gd2f=None)[0]

            if dgdf is None:
                dgdx = torch.eye(self.width, dtype=dgdx.dtype, device=dgdx.device) + self.h * dgdx
            else:
                dgdx = dgdf + self.h * dgdx
        else:
            dfdx, d2fd2x = self.layer.backward(do_Hessian=do_Hessian, dgdf=None, d2gd2f=None)[:2]

            dgdx = torch.eye(self.width, dtype=dfdx.dtype, device=dfdx.device) + self.h * dfdx
            if dgdf is not None:
                dgdx = dgdx @ dgdf

            d2gd2x = self.h * d2fd2x
            if d2gd2f is not None:
                # TODO: compare timings for h_dfdx
                h_dfdx = torch.eye(self.width, dtype=dfdx.dtype, device=dfdx.device) + self.h * dfdx

                # h_dfdx = self.h * dfdx
                # idx = torch.arange(dfdx.shape[-1])
                # h_dfdx[:, idx, idx] += 1.0

                # Gauss-Newton approximation
                h1 = (h_dfdx.unsqueeze(1) @ d2gd2f.permute(0, 3, 1, 2) @ h_dfdx.permute(0, 2, 1).unsqueeze(1))
                h1 = h1.permute(0, 2, 3, 1)

                # extra term to compute full Hessian
                N, _, _, m = d2fd2x.shape
                h2 = d2fd2x.reshape(N, -1, m) @ dgdf.reshape(N, m, -1)
                h2 = self.h * h2.reshape(h1.shape)

                # combine
                d2gd2x = h1 + h2

        return dgdx, d2gd2x

    def extra_repr(self) -> str:
        return 'width={}, h={}'.format(self.width, self.h)
