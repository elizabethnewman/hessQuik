import torch
import torch.nn as nn
import math


class quadraticLayer(nn.Module):
    """
    f(x) = x @ v + 0.5 * x.t() @ A.t() @ A @ x + mu
    """

    def __init__(self, in_features, rank):
        super(quadraticLayer, self).__init__()

        self.in_features = in_features
        self.rank = rank
        self.ctx = None

        # create final layer
        self.v = nn.Parameter(torch.empty(self.in_features))
        self.mu = nn.Parameter(torch.empty(1))
        self.A = nn.Parameter(torch.empty(self.rank, self.in_features))
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.v, a=-bound, b=bound)
        nn.init.uniform_(self.mu)
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.A, a=-bound, b=bound)

    def forward(self, u, do_gradient=False, do_Hessian=False, dudx=None, d2ud2x=None, reverse_mode=False):

        (df, d2f) = (None, None)
        AtA = self.A.t() @ self.A
        f = u @ self.v + 0.5 * torch.sum((u @ AtA) * u, dim=1) + self.mu

        if reverse_mode:
            self.ctx = (u,)

        if (do_gradient or do_Hessian) and not reverse_mode:
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

        return f.unsqueeze(-1), df, d2f

    def backward(self, do_Hessian=False, dgdf=None, d2gd2f=None):
        d2f = None

        x = self.ctx[0]
        AtA = self.A.t() @ self.A  # TODO: recompute this or store it?
        df = self.v.unsqueeze(0) + torch.matmul(x, AtA)

        if do_Hessian:
            # TODO: improve wasteful storage
            d2f = (torch.ones(x.shape[0], 1, 1, dtype=AtA.dtype, device=AtA.device) * AtA).unsqueeze(-1)

        return df.unsqueeze(-1), d2f


if __name__ == '__main__':
    from hessQuik.tests.utils import DerivativeCheckTestsNetwork
    torch.set_default_dtype(torch.float64)

    # problem setup
    nex = 11  # no. of examples
    d = 4  # no. of input dimensiona features
    m = 7  # rank
    x = torch.randn(nex, d)
    dx = torch.randn_like(x)
    f = quadraticLayer(d, m)

    # forward tests
    derivativeTests = DerivativeCheckTestsNetwork()

    print('======= FORWARD =======')
    derivativeTests.run_forward_hessian_test(f, x, dx, verbose=True)

    print('======= BACKWARD =======')
    derivativeTests.run_backward_hessian_test(f, x, dx, verbose=True)

