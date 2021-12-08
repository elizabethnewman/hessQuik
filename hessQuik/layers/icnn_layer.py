import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from hessQuik.layers.layer_types import hessQuikLayer
import hessQuik.layers.activation_functions as act


class ICNNLayer(hessQuikLayer):
    """
    layer of an input convex neural network f : R^{min} x R^d \to R^{mout} where

        f(u,x) = act(z(u,x)) with z(u,x) =  m(L)*u + K*x + b

    with implementation its input gradients and Hessians

    Properties:
        d           - no. of input features, i.e., size of x
        min         - no. of auxiliary input features, i.e., size of u
        mout        - no. of output features
        act         - activation function, default=softplus
        m           - element-wise scaling function to ensure non-negativity of some weights, default=softplus
        K           - weights applied to x, K.shape=(d,mout)
        L           - weights applied to u, L.shape=(min,mout)
        b           - weights for bias, b.shape=mout

    """

    def __init__(self, input_dim, in_features, out_features, act: act.activationFunction = act.softplusActivation(),
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(ICNNLayer, self).__init__()

        self.input_dim = input_dim
        self.in_features = in_features
        self.out_features = out_features
        self.act = act

        # extract nonnegative weights
        self.nonneg = F.softplus

        self.K = nn.Parameter(torch.empty(input_dim, out_features, **factory_kwargs))

        if in_features is not None:
            self.L = nn.Parameter(torch.empty(in_features, out_features, **factory_kwargs))
        else:
            self.register_parameter('L', None)

        self.b = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.K, a=math.sqrt(self.input_dim))

        if self.L is not None:
            nn.init.kaiming_uniform_(self.L, a=math.sqrt(self.in_features))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.L)
        else:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.K)

        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.b, -bound, bound)

    def forward(self, ux, do_gradient=False, do_Hessian=False, dudx=None, d2ud2x=None, reverse_mode=False):

        (dfdx, d2fd2x) = (None, None)

        M = self.K
        if self.L is not None:
            M = torch.cat((self.nonneg(self.L), M), dim=0)

        z = ux @ M + self.b

        # forward pass
        f, dsig, d2sig = self.act.forward(z, do_gradient=do_gradient, do_Hessian=do_Hessian, reverse_mode=reverse_mode)
        f = torch.cat((f, ux[:, -self.input_dim:]), dim=1)

        if (do_gradient or do_Hessian) and not reverse_mode:
            dfdx = dsig.unsqueeze(1) * M

            # -------------------------------------------------------------------------------------------------------- #
            if do_Hessian:
                d2fd2x = (d2sig.unsqueeze(1) * M).unsqueeze(2) * M.unsqueeze(0).unsqueeze(0)

                # Gauss-Newton approximation
                if dudx is not None:
                    d2fd2x = dudx.unsqueeze(1) @ (d2fd2x.permute(0, 3, 1, 2) @ dudx.unsqueeze(1).permute(0, 1, 3, 2))
                    d2fd2x = d2fd2x.permute(0, 2, 3, 1)

                if d2ud2x is not None:
                    # extra term to compute full Hessian
                    d2fd2x += d2ud2x @ dfdx.unsqueeze(1)

                # concatenate zeros
                Z = torch.zeros(d2fd2x.shape[0], d2fd2x.shape[1], d2fd2x.shape[2], self.input_dim,
                                dtype=d2fd2x.dtype, device=d2fd2x.device)
                d2fd2x = torch.cat((d2fd2x, Z), dim=-1)
            # -------------------------------------------------------------------------------------------------------- #

            # finish computing gradient
            if dudx is not None:
                dfdx = dudx @ dfdx

            I = torch.ones(dfdx.shape[0], 1, 1, dtype=dfdx.dtype, device=dfdx.device) \
                * torch.eye(self.input_dim, dtype=dfdx.dtype, device=dfdx.device).unsqueeze(0)
            dfdx = torch.cat((dfdx, I), dim=-1)

        return f, dfdx, d2fd2x

    def backward(self, do_Hessian=False, dgdf=None, d2gd2f=None):

        M = self.K
        if self.L is not None:
            M = torch.cat((self.nonneg(self.L), M), dim=0)

        # obtain stored information from backward pass
        d2gd2x = None
        dsig, d2sig = self.act.backward(do_Hessian=do_Hessian)

        # compute gradient
        dgdx = dsig.unsqueeze(1) * M

        # augment gradient
        M2 = torch.ones(dgdx.shape[0], 1, 1, dtype=dgdx.dtype, device=dgdx.device) \
            * torch.eye(self.input_dim, dtype=dgdx.dtype, device=dgdx.device).unsqueeze(0)

        if self.in_features is not None:
            Z = torch.zeros(dgdx.shape[0], self.input_dim, self.in_features)
            M2 = torch.cat((Z, M2), dim=-1).permute(0, 2, 1)

        dgdx = torch.cat((dgdx, M2), dim=-1)

        if do_Hessian:
            # TODO: change order of operations, multiply K's first; check if logic with better naming
            d2gd2x = (d2sig.unsqueeze(1) * M.unsqueeze(0)).unsqueeze(2) * M.unsqueeze(0).unsqueeze(0)

            # concatenate zeros
            Z = torch.zeros(d2gd2x.shape[0], d2gd2x.shape[1], d2gd2x.shape[2], self.input_dim,
                            dtype=d2gd2x.dtype, device=d2gd2x.device)
            d2gd2x = torch.cat((d2gd2x, Z), dim=-1)

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
