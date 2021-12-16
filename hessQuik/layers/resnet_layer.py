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
                 device=None, dtype=None):
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
        dgdx, d2gd2x = self.layer.backward(do_Hessian=do_Hessian, dgdf=dgdf,
                                           d2gd2f=self.h * d2gd2f if d2gd2f is not None else None)

        if do_Hessian:
            d2gd2x *= self.h
            if d2gd2f is not None:
                dsig, _ = self.layer.act.backward(do_Hessian=False)  # TODO: this is computed in layer.backward
                h3 = (dsig.unsqueeze(1) * self.layer.K).unsqueeze(1) @ (self.h * d2gd2f)
                d2gd2x += d2gd2f + (h3 + h3.permute(0, 2, 1, 3))

        # finish computing gradient
        dgdx *= self.h
        if dgdf is None:
            dgdx += torch.eye(self.width, dtype=dgdx.dtype, device=dgdx.device)
        else:
            dgdx += dgdf

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

    print('======= FORWARD =======')
    input_derivative_check(f, x, do_Hessian=True, verbose=True, forward_mode=True)

    print('======= BACKWARD =======')
    input_derivative_check(f, x, do_Hessian=True, verbose=True, forward_mode=False)