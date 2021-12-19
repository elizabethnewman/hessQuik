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

    def dim_input(self):
        return self.in_features

    def dim_output(self):
        return self.out_features

    def forward(self, u, do_gradient=False, do_Hessian=False, forward_mode=True, dudx=None, d2ud2x=None):

        (dfdx, d2fd2x) = (None, None)
        f, dsig, d2sig = self.act(u @ self.K + self.b, do_gradient=do_gradient, do_Hessian=do_Hessian,
                                  forward_mode=True if forward_mode is True else None)
        # ------------------------------------------------------------------------------------------------------------ #
        # forward mode
        if (do_gradient or do_Hessian) and forward_mode is True:
            dfdx = dsig.unsqueeze(1) * self.K
            # -------------------------------------------------------------------------------------------------------- #
            if do_Hessian:
                d2fd2x = (d2sig.unsqueeze(1) * self.K).unsqueeze(2) * self.K.unsqueeze(0).unsqueeze(0)

                # Gauss-Newton approximation
                if d2ud2x is not None:
                    d2fd2x = dudx.unsqueeze(1) @ (d2fd2x.permute(0, 3, 1, 2) @ dudx.permute(0, 2, 1).unsqueeze(1))
                    d2fd2x = d2fd2x.permute(0, 2, 3, 1)

                    # extra term to compute full Hessian
                    d2fd2x += d2ud2x @ dfdx.unsqueeze(1)
            # -------------------------------------------------------------------------------------------------------- #
            # finish computing gradient
            if dudx is not None:
                dfdx = dudx @ dfdx

        # ------------------------------------------------------------------------------------------------------------ #
        # backward mode (if layer is not wrapped in NN)
        if (do_gradient or do_Hessian) and forward_mode is False:
            dfdx, d2fd2x = self.backward(do_Hessian=do_Hessian)

        return f, dfdx, d2fd2x

    def backward(self, do_Hessian=False, dgdf=None, d2gd2f=None):
        d2gd2x = None
        dsig, d2sig = self.act.backward(do_Hessian=do_Hessian)
        dgdx = dsig.unsqueeze(1) * self.K

        if do_Hessian:
            d2gd2x = (d2sig.unsqueeze(1) * self.K.unsqueeze(0)).unsqueeze(2) * self.K.unsqueeze(0).unsqueeze(0)

            if d2gd2f is not None:
                # Gauss-Newton approximation
                h1 = (dgdx.unsqueeze(1) @ d2gd2f.permute(0, 3, 1, 2) @ dgdx.permute(0, 2, 1).unsqueeze(1))
                h1 = h1.permute(0, 2, 3, 1)

                # extra term to compute full Hessian
                h2 = d2gd2x @ dgdf.unsqueeze(1)
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


if __name__ == '__main__':
    from hessQuik.utils import input_derivative_check
    torch.set_default_dtype(torch.float64)

    nex = 11  # no. of examples
    d = 4  # no. of input features
    m = 7  # no. of output features
    x = torch.randn(nex, d)

    f = singleLayer(d, m, act=act.softplusActivation())

    print('======= FORWARD =======')
    input_derivative_check(f, x, do_Hessian=True, verbose=True, forward_mode=True)

    print('======= BACKWARD =======')
    input_derivative_check(f, x, do_Hessian=True, verbose=True, forward_mode=False)
