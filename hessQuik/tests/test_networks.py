
import unittest
import torch
import hessQuik.activations as act
import hessQuik.layers as lay
from hessQuik.networks import NN
from hessQuik.tests.utils import DerivativeCheckTestsNetwork


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

    @staticmethod
    def run_test(f, x, dx, requires_grad=False):
        x.requires_grad = requires_grad

        derivativeTests = DerivativeCheckTestsNetwork()

        # forward tests
        derivativeTests.run_forward_gradient_test(f, x, dx)
        derivativeTests.run_forward_hessian_test(f, x, dx)

        derivativeTests.run_backward_gradient_test(f, x, dx)
        derivativeTests.run_backward_hessian_test(f, x, dx)

    def test_NN_scalar_output(self):
        # problem setup
        f, x = self.setup_network(1)
        dx = torch.randn_like(x)

        print(self, ': scalar output')
        self.run_test(f, x, dx, requires_grad=False)

    def test_NN_vector_output(self):
        # problem setup
        m = 3
        f, x = self.setup_network(m)
        dx = torch.randn_like(x)

        print(self, ': vector output')
        self.run_test(f, x, dx, requires_grad=False)


class TestICNNNetwork(unittest.TestCase):

    @staticmethod
    def setup_network(m):
        nex = 11  # no. of examples
        d = 3  # no. of input features
        ms = [None, 5, 2, 7]  # no. of output features
        x = torch.randn(nex, d)

        f = NN(lay.ICNNLayer(d, ms[0], ms[1], act=act.softplusActivation()),
               lay.ICNNLayer(d, ms[1], ms[2], act=act.softplusActivation()),
               lay.ICNNLayer(d, ms[2], ms[3], act=act.softplusActivation()))

        return f, x

    @staticmethod
    def run_test(f, x, dx, requires_grad=False):
        x.requires_grad = requires_grad

        derivativeTests = DerivativeCheckTestsNetwork()

        # forward tests
        derivativeTests.run_forward_gradient_test(f, x, dx, verbose=False)
        derivativeTests.run_forward_hessian_test(f, x, dx)
        derivativeTests.run_backward_gradient_test(f, x, dx)
        derivativeTests.run_backward_hessian_test(f, x, dx)

    def test_ICNN(self):
        f, x = self.setup_network(1)
        dx = torch.randn_like(x)

        print(self, ': vector output')
        self.run_test(f, x, dx, requires_grad=False)


if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    unittest.main()