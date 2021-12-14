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
        self.reverse_mode = reverse_mode

    def dim_input(self):
        return self.width

    def dim_output(self):
        return self.width

    @property
    def reverse_mode(self):
        return self._reverse_mode

    @reverse_mode.setter
    def reverse_mode(self, reverse_mode):
        self._reverse_mode = reverse_mode
        self.layer.reverse_mode = reverse_mode

    def forward(self, u, do_gradient=False, do_Hessian=False, do_Laplacian=False, dudx=None, d2ud2x=None, lap_u=None):

        (dfdx, d2fd2x, lap_f) = (None, None, None)
        fi, dfi, d2fi, lap_fi = self.layer(u, do_gradient=do_gradient, do_Hessian=do_Hessian, do_Laplacian=do_Laplacian,
                                           dudx=dudx, d2ud2x=d2ud2x, lap_u=lap_u)

        # skip connection
        f = u + self.h * fi

        if do_gradient and self.reverse_mode is False:

            if dudx is None:
                dfdx = torch.eye(self.width, dtype=dfi.dtype, device=dfi.device) + self.h * dfi
            else:
                dfdx = dudx + self.h * dfi

        if do_Hessian and self.reverse_mode is False:
            d2fd2x = self.h * d2fi
            if d2ud2x is not None:
                d2fd2x += d2ud2x

        if (do_gradient or do_Hessian) and self.reverse_mode is True:
            dfdx, d2fd2x, lap_f = self.backward(do_Hessian=do_Hessian, do_Laplacian=do_Laplacian)

        return f, dfdx, d2fd2x, lap_f

    def backward(self, do_Hessian=False, do_Laplacian=False, dgdf=None, d2gd2f=None, lap_g=None):
        d2gd2x = None
        if not do_Hessian:

            dgdx = self.layer.backward(do_Hessian=False, do_Laplacian=do_Laplacian,
                                       dgdf=dgdf, d2gd2f=None, lap_g=None)[0]

            if dgdf is None:
                dgdx = torch.eye(self.width, dtype=dgdx.dtype, device=dgdx.device) + self.h * dgdx
            else:
                dgdx = dgdf + self.h * dgdx
        else:
            dfdx, d2fd2x = self.layer.backward(do_Hessian=do_Hessian, do_Laplacian=do_Laplacian,
                                               dgdf=None, d2gd2f=None, lap_g=None)[:2]

            dgdx = torch.eye(self.width, dtype=dfdx.dtype, device=dfdx.device) + self.h * dfdx
            if dgdf is not None:
                dgdx = dgdx @ dgdf

            d2gd2x = self.h * d2fd2x
            if d2gd2f is not None:
                # TODO: compare timings for h_dfdx on CPU and GPU
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

        return dgdx, d2gd2x, lap_g

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

    print('======= FORWARD =======')
    f.reverse_mode = False
    input_derivative_check(f, x, do_Hessian=True, verbose=True)

    print('======= BACKWARD =======')
    f.reverse_mode = True
    input_derivative_check(f, x, do_Hessian=True, verbose=True)