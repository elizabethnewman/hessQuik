import torch
import torch.nn as nn
import math
from hessQuik.layers import hessQuikLayer


class quadraticLayer(hessQuikLayer):
    """
    f(x) = x @ v + 0.5 * x.t() @ A.t() @ A @ x + mu
    """

    def __init__(self, in_features, rank, device=None, dtype=None, reverse_mode=False):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(quadraticLayer, self).__init__()

        self.in_features = in_features
        self.rank = rank
        self.ctx = None
        self.reverse_mode = reverse_mode

        # create final layer
        self.v = nn.Parameter(torch.empty(self.in_features, **factory_kwargs))
        self.mu = nn.Parameter(torch.empty(1, **factory_kwargs))
        self.A = nn.Parameter(torch.empty(self.rank, self.in_features, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.v, a=-bound, b=bound)
        nn.init.uniform_(self.mu)
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.A, a=-bound, b=bound)

    def dim_input(self):
        return self.in_features

    def dim_output(self):
        return 1

    @property
    def reverse_mode(self):
        return self._reverse_mode

    @reverse_mode.setter
    def reverse_mode(self, reverse_mode):
        self._reverse_mode = reverse_mode

    def forward(self, u, do_gradient=False, do_Hessian=False, dudx=None, d2ud2x=None):

        (df, d2f) = (None, None)
        AtA = self.A.t() @ self.A
        f = u @ self.v + 0.5 * torch.sum((u @ AtA) * u, dim=1) + self.mu

        if self.reverse_mode is not False:
            self.ctx = (u,)

        if (do_gradient or do_Hessian) and self.reverse_mode is False:
            print('here1')
            df = self.v.unsqueeze(0) + u @ AtA

            if do_Hessian:
                d2f = AtA

                if d2ud2x is not None:
                    d2f = dudx @ d2f @ dudx.permute(0, 2, 1)
                    d2f += (d2ud2x @ df.unsqueeze(1).unsqueeze(-1)).squeeze()

                d2f = d2f.unsqueeze(-1)

            # finish computing gradient
            df = df.unsqueeze(-1)

            if dudx is not None:
                df = dudx @ df

        if (do_gradient or do_Hessian) and self.reverse_mode is True:
            df, d2f = self.backward(do_Hessian=do_Hessian)

        return f.unsqueeze(-1), df, d2f

    def backward(self, do_Hessian=False, dgdf=None, d2gd2f=None):
        print('here2')
        d2f = None

        x = self.ctx[0]
        AtA = self.A.t() @ self.A  # TODO: recompute this or store it?
        df = self.v.unsqueeze(0) + torch.matmul(x, AtA)

        if do_Hessian:
            # TODO: improve wasteful storage
            d2f = (torch.ones(x.shape[0], 1, 1, dtype=AtA.dtype, device=AtA.device) * AtA).unsqueeze(-1)

        return df.unsqueeze(-1), d2f


if __name__ == '__main__':
    from hessQuik.utils import input_derivative_check
    torch.set_default_dtype(torch.float64)

    # problem setup
    nex = 11  # no. of examples
    d = 4  # no. of input dimensiona features
    m = 7  # rank
    x = torch.randn(nex, d)
    f = quadraticLayer(d, m)

    print('======= FORWARD =======')
    f.reverse_mode = False
    input_derivative_check(f, x, do_Hessian=True, verbose=True)

    print('======= BACKWARD =======')
    f.reverse_mode = True
    input_derivative_check(f, x, do_Hessian=True, verbose=True)


