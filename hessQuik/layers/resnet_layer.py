import torch
from hessQuik.layers.layer_types import hessQuikLayer
import hessQuik.layers.activation_functions as act
from hessQuik.layers.single_layer import singleLayer


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

    def __init__(self, width, h=1.0, act: act.activationFunction = act.identityActivation(), device=None, dtype=None):
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
