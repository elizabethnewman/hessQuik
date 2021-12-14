import torch
import torch.nn as nn
import math
from hessQuik.layers import hessQuikLayer
import hessQuik.activations as act


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

    def __init__(self, in_features, out_features, act: act.hessQuikActivationFunction = act.identityActivation(),
                 device=None, dtype=None, reverse_mode=False):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(singleLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.act = act
        self.reverse_mode = reverse_mode

        self.K = nn.Parameter(torch.empty(in_features, out_features, **factory_kwargs))
        self.b = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.K, a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.K)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.b, -bound, bound)

    def dim_input(self):
        return self.in_features

    def dim_output(self):
        return self.out_features

    @property
    def reverse_mode(self):
        return self._reverse_mode

    @reverse_mode.setter
    def reverse_mode(self, reverse_mode):
        self._reverse_mode = reverse_mode
        self.act.reverse_mode = False if reverse_mode is False else None

    def forward(self, u, do_gradient=False, do_Hessian=False, do_Laplacian=False, dudx=None, d2ud2x=None, lap_u=None):

        (dfdx, d2fd2x, lap_f) = (None, None, None)
        f, dsig, d2sig = self.act.forward(u @ self.K + self.b,
                                          do_gradient=do_gradient, do_Hessian=do_Hessian, do_Laplacian=do_Laplacian)

        # ------------------------------------------------------------------------------------------------------------ #
        # forward mode
        if (do_gradient or do_Hessian or do_Laplacian) and self.reverse_mode is False:
            dfdx = dsig.unsqueeze(1) * self.K
            # -------------------------------------------------------------------------------------------------------- #
            if do_Hessian:
                d2fd2x = (d2sig.unsqueeze(1) * self.K).unsqueeze(2) * self.K.unsqueeze(0).unsqueeze(0)

                # TODO: compare alternative computation - roughly the same amount of time to compute
                # d2fd2x = (d2sig.unsqueeze(-1).unsqueeze(-1) * (self.K.T.unsqueeze(-1) @ self.K.T.unsqueeze(1)))
                # d2fd2x = d2fd2x.permute(0, 2, 3, 1)

                # Gauss-Newton approximation
                if dudx is not None:
                    d2fd2x = dudx.unsqueeze(1) @ (d2fd2x.permute(0, 3, 1, 2) @ dudx.unsqueeze(1).permute(0, 1, 3, 2))
                    d2fd2x = d2fd2x.permute(0, 2, 3, 1)

                if d2ud2x is not None:
                    # extra term to compute full Hessian
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
            # -------------------------------------------------------------------------------------------------------- #
            # finish computing gradient
            if dudx is not None:
                dfdx = dudx @ dfdx

        # ------------------------------------------------------------------------------------------------------------ #
        # backward mode (if layer is not wrapped in NN)
        if (do_gradient or do_Hessian or do_Laplacian) and self.reverse_mode is True:
            dfdx, d2fd2x, lap_f = self.backward(do_Hessian=do_Hessian, do_Laplacian=do_Laplacian)

        return f, dfdx, d2fd2x, lap_f

    def backward(self, do_Hessian=False, do_Laplacian=False, dgdf=None, d2gd2f=None, lap_g=None):
        (d2gd2x, lap_g) = (None, None)
        dsig, d2sig = self.act.backward(do_Hessian=do_Hessian, do_Laplacian=do_Laplacian)
        dgdx = dsig.unsqueeze(1) * self.K

        if do_Hessian:
            d2gd2x = (d2sig.unsqueeze(1) * self.K.unsqueeze(0)).unsqueeze(2) * self.K.unsqueeze(0).unsqueeze(0)

            # TODO: compare alternative computation - roughly the same amount of time to compute
            # d2gd2x = (d2sig.unsqueeze(-1).unsqueeze(-1) * (self.K.T.unsqueeze(-1) @ self.K.T.unsqueeze(1)))
            # d2gd2x = d2gd2x.permute(0, 2, 3, 1)

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

        return dgdx, d2gd2x, lap_g

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )


if __name__ == '__main__':
    from hessQuik.utils import input_derivative_check, laplcian_check_using_hessian
    torch.set_default_dtype(torch.float64)

    nex = 11  # no. of examples
    d = 4  # no. of input features
    m = 7  # no. of output features
    x = torch.randn(nex, d)

    print('======= FORWARD =======')
    f = singleLayer(d, m, act=act.softplusActivation())
    f.reverse_mode = False
    input_derivative_check(f, x, do_Hessian=True, verbose=True)

    print('======= BACKWARD =======')
    f = singleLayer(d, m, act=act.softplusActivation())
    f.reverse_mode = True
    input_derivative_check(f, x, do_Hessian=True, verbose=True)

    print('======= LAPLACIAN: FORWARD =======')
    f.reverse_mode = False
    laplcian_check_using_hessian(f, x)

    # print('======= LAPLACIAN: BACKWARD =======')
    # f.reverse_mode = True
    # laplcian_check_using_hessian(f, x)
