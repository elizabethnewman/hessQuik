import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from hessQuik.layers import hessQuikLayer


class quadraticICNNLayer(hessQuikLayer):
    """
    f(x) = u @ nonneg(w) + x @ v + 0.5 * x.t() @ A.t() @ A @ x + mu
    """

    def __init__(self, input_dim, in_features, rank, device=None, dtype=None, reverse_mode=False):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(quadraticICNNLayer, self).__init__()

        self.input_dim = input_dim
        self.in_features = in_features
        self.rank = rank
        self.ctx = None
        self.nonneg = F.softplus
        self.reverse_mode = reverse_mode

        # create final layer

        if in_features is not None:
            self.w = nn.Parameter(torch.empty(in_features, **factory_kwargs))
        else:
            self.register_parameter('w', None)

        self.v = nn.Parameter(torch.empty(input_dim, **factory_kwargs))
        self.mu = nn.Parameter(torch.empty(1, **factory_kwargs))
        self.A = nn.Parameter(torch.empty(rank, input_dim, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self):

        if self.in_features is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.w, a=-bound, b=bound)
        else:
            bound = 1 / math.sqrt(self.input_dim)

        nn.init.uniform_(self.v, a=-bound, b=bound)
        nn.init.uniform_(self.mu)
        bound = 1 / math.sqrt(self.input_dim)
        nn.init.uniform_(self.A, a=-bound, b=bound)

    def dim_input(self):
        return self.in_features + self.input_dim

    def dim_output(self):
        return 1

    @property
    def reverse_mode(self):
        return self._reverse_mode

    @reverse_mode.setter
    def reverse_mode(self, reverse_mode):
        self._reverse_mode = reverse_mode

    def forward(self, ux, do_gradient=False, do_Hessian=False, dudx=None, d2ud2x=None):

        (df, d2f) = (None, None)
        AtA = self.A.t() @ self.A

        if self.w is None:
            w = torch.empty(0, dtype=self.v.dtype, device=self.v.device)
        else:
            w = self.nonneg(self.w)

        wv = torch.cat((w, self.v), dim=0)
        x = ux[:, -self.input_dim:]

        # forward propagate
        f = ux @ wv + 0.5 * torch.sum((x @ AtA) * x, dim=1) + self.mu

        if self.reverse_mode is not False:
            self.ctx = (ux,)

        # ------------------------------------------------------------------------------------------------------------ #
        if (do_gradient or do_Hessian) and self.reverse_mode is False:
            print('here1')
            if self.in_features is None:
                z = torch.empty(ux.shape[0], 0)
            else:
                z = torch.zeros(ux.shape[0], self.in_features)

            df = wv.unsqueeze(0) + torch.cat((z, x @ AtA), dim=1)

            # -------------------------------------------------------------------------------------------------------- #
            if do_Hessian:
                d2f = AtA

                if d2ud2x is not None:
                    d2f = dudx[:, :, -self.input_dim:] @ d2f @ dudx[:, :, -self.input_dim:].permute(0, 2, 1)
                    z = torch.zeros(x.shape[0], self.in_features)
                    d2f += (d2ud2x @ (
                                torch.cat((w, self.v), dim=0).unsqueeze(0) + torch.cat((z, x @ AtA), dim=1)).unsqueeze(
                        1).unsqueeze(-1)).squeeze()

                d2f = d2f.unsqueeze(-1)
            # -------------------------------------------------------------------------------------------------------- #
            # finish computing gradient
            if dudx is not None:
                df = (dudx @ df.unsqueeze(-1)).squeeze()

            df = df.unsqueeze(-1)

        if (do_gradient or do_Hessian) and self.reverse_mode is True:
            df, d2f = self.backward(do_Hessian=do_Hessian)

        return f.unsqueeze(-1), df, d2f

    def backward(self, do_Hessian=False, dgdf=None, d2gd2f=None):
        print('here2')
        d2f = None

        ux = self.ctx[0]
        x = ux[:, -self.input_dim:]
        AtA = self.A.t() @ self.A  # TODO: recompute this or store it?

        wv = self.v
        if self.w is not None:
            wv = torch.cat((self.nonneg(self.w), wv), dim=0)

        z = torch.empty(ux.shape[0], 0)
        if self.in_features is not None:
            z = torch.zeros(ux.shape[0], self.in_features)

        df = wv.unsqueeze(0) + torch.cat((z, x @ AtA), dim=1)

        if do_Hessian:
            e = torch.ones(x.shape[0], 1, 1, dtype=AtA.dtype, device=AtA.device)
            d2f = torch.zeros(x.shape[0], ux.shape[1], ux.shape[1])
            d2f[:, -self.input_dim:, -self.input_dim:] = e * AtA
            d2f = d2f.unsqueeze(-1)

        return df.unsqueeze(-1), d2f


if __name__ == '__main__':
    from hessQuik.utils import input_derivative_check
    torch.set_default_dtype(torch.float64)

    nex = 11  # no. of examples
    d = 4  # no. of input features
    in_feat = 5
    m = 13  # rank
    x = torch.randn(nex, d)
    f = quadraticICNNLayer(d, None, m)

    print('======= FORWARD =======')
    f.reverse_mode = False
    input_derivative_check(f, x, do_Hessian=True, verbose=True)

    print('======= BACKWARD =======')
    f.reverse_mode = True
    input_derivative_check(f, x, do_Hessian=True, verbose=True)
