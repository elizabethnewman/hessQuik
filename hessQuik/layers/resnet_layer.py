import torch
from hessQuik.layers import hessQuikLayer
import hessQuik.activations as act
from hessQuik.layers import singleLayer


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

    def __init__(self, width, h=1.0, act: act.hessQuikActivationFunction = act.identityActivation(),
                 device=None, dtype=None, reverse_mode=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(resnetLayer, self).__init__()

        self.width = width
        self.h = h
        self.layer = singleLayer(width, width, act=act, **factory_kwargs)

    def dim_input(self):
        return self.width

    def dim_output(self):
        return self.width

    def forward(self, u, do_gradient=False, do_Hessian=False, forward_mode=True, dudx=None, d2ud2x=None):

        (dfdx, d2fd2x) = (None, None)
        fi, dfi, d2fi = self.layer(u, do_gradient=do_gradient, do_Hessian=do_Hessian, dudx=dudx, d2ud2x=d2ud2x,
                                   forward_mode=True if forward_mode is True else None)

        # skip connection
        f = u + self.h * fi

        if do_gradient and forward_mode is True:

            if dudx is None:
                dfdx = torch.eye(self.width, dtype=dfi.dtype, device=dfi.device) + self.h * dfi
            else:
                dfdx = dudx + self.h * dfi

        if do_Hessian and forward_mode is True:
            d2fd2x = self.h * d2fi
            if d2ud2x is not None:
                d2fd2x += d2ud2x

        if (do_gradient or do_Hessian) and forward_mode is False:
            dfdx, d2fd2x = self.backward(do_Hessian=do_Hessian)

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
                # TODO: compare timings for h_dfdx on CPU and GPU
                h_dfdx = torch.eye(self.width, dtype=dfdx.dtype, device=dfdx.device) + self.h * dfdx

                # Gauss-Newton approximation
                h1 = (h_dfdx.unsqueeze(1) @ d2gd2f.permute(0, 3, 1, 2) @ h_dfdx.permute(0, 2, 1).unsqueeze(1))
                h1 = h1.permute(0, 2, 3, 1)

                # extra term to compute full Hessian
                h2 = d2fd2x @ dgdf.unsqueeze(1)
                # combine
                d2gd2x = h1 + h2

        return dgdx, d2gd2x

    def extra_repr(self) -> str:
        return 'width={}, h={}'.format(self.width, self.h)


if __name__ == '__main__':
    from hessQuik.utils import input_derivative_check
    torch.set_default_dtype(torch.float64)

    nex = 11  # no. of examples
    width = 4  # no. of input features
    h = 0.25
    x = torch.randn(nex, width)
    f = resnetLayer(width, h=h, act=act.softplusActivation())

    f0, df0, d2f0 = f(x, do_gradient=True, do_Hessian=True, forward_mode=False)

    # print('======= FORWARD =======')
    # input_derivative_check(f, x, do_Hessian=True, verbose=True, forward_mode=True)
    #
    # print('======= BACKWARD =======')
    # input_derivative_check(f, x, do_Hessian=True, verbose=True, forward_mode=False)