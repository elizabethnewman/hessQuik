
import unittest
import torch
import hessQuik.activations as act
import hessQuik.layers as lay
from hessQuik.networks import NN, fullyConnectedNN, resnetNN, ICNN
from hessQuik.tests.utils import DerivativeCheckTestsNetwork


def run_test(f, x, dx, requires_grad=False):
    x.requires_grad = requires_grad

    derivativeTests = DerivativeCheckTestsNetwork()

    # forward tests
    derivativeTests.run_forward_gradient_test(f, x, dx)
    derivativeTests.run_forward_hessian_test(f, x, dx)

    derivativeTests.run_backward_gradient_test(f, x, dx)
    derivativeTests.run_backward_hessian_test(f, x, dx)


class TestNetwork(unittest.TestCase):

    @staticmethod
    def setup_network(m):
        nex = 11
        d = 3
        ms = [2, 7, 5]
        x = torch.randn(nex, d)

        f = NN(lay.singleLayer(d, ms[0], act=act.softplusActivation()),
               lay.singleLayer(ms[0], ms[1], act=act.softplusActivation()),
               lay.resnetLayer(ms[1], h=0.25, act=act.antiTanhActivation()),
               lay.singleLayer(ms[1], ms[2], act=act.softplusActivation()),
               lay.resnetLayer(ms[2], h=0.5, act=act.quadraticActivation()),
               lay.singleLayer(ms[2], m, act=act.softplusActivation()))

        return f, x

    def test_NN_scalar_output(self):
        # problem setup
        f, x = self.setup_network(1)
        dx = torch.randn_like(x)

        print(self, ': scalar output')
        run_test(f, x, dx, requires_grad=False)

    def test_NN_vector_output(self):
        # problem setup
        m = 3
        f, x = self.setup_network(m)
        dx = torch.randn_like(x)

        print(self, ': vector output')
        run_test(f, x, dx, requires_grad=False)


class TestFullyConnectedNN(unittest.TestCase):

    def test_fullyConnectedNN(self):
        nex = 11
        d = 3
        ms = [d, 2, 7, 5]
        x = torch.randn(nex, d)

        f = fullyConnectedNN(ms, act=act.antiTanhActivation())
        dx = torch.randn_like(x)

        print(self)
        run_test(f, x, dx, requires_grad=False)


class TestResnetNN(unittest.TestCase):

    def test_resnetNN(self):
        nex = 11
        d = 3
        x = torch.randn(nex, d)

        f = resnetNN(d, 4, h=0.25, act=act.quadraticActivation())
        dx = torch.randn_like(x)

        print(self)
        run_test(f, x, dx, requires_grad=False)


class TestICNNNetwork(unittest.TestCase):

    def test_ICNNNetworkLayers(self):
        nex = 11  # no. of examples
        d = 3  # no. of input features
        ms = [None, 5, 2, 7]  # no. of output features

        f = NN(lay.ICNNLayer(d, ms[0], ms[1], act=act.softplusActivation()),
               lay.ICNNLayer(d, ms[1], ms[2], act=act.softplusActivation()),
               lay.ICNNLayer(d, ms[2], ms[3], act=act.softplusActivation()))

        x = torch.randn(nex, d)
        dx = torch.randn_like(x)

        print(self)
        run_test(f, x, dx, requires_grad=False)

    def test_ICNN(self):
        nex = 11  # no. of examples
        d = 3  # no. of input features
        ms = [None, 5, 2, 7]  # no. of output features

        f = ICNN(d, ms, act=act.softplusActivation())

        x = torch.randn(nex, d)
        dx = torch.randn_like(x)

        print(self)
        run_test(f, x, dx, requires_grad=False)


if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    unittest.main()
