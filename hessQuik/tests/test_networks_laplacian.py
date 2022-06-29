import unittest
import torch
import hessQuik.activations as act
import hessQuik.layers as lay
import hessQuik.networks as net
from hessQuik.tests.utils import run_all_tests_laplacian


class TestNN(unittest.TestCase):

    @staticmethod
    def setup_network(m):
        torch.set_default_dtype(torch.float64)
        nex = 11
        d = 3
        ms = [2, 7, 5]
        x = torch.randn(nex, d)

        f = net.NN(lay.singleLayer(d, ms[0], act=act.softplusActivation()),
                   lay.singleLayer(ms[0], ms[1], act=act.softplusActivation()),
                   lay.resnetLayer(ms[1], h=0.25, act=act.antiTanhActivation()),
                   lay.singleLayer(ms[1], ms[2], act=act.softplusActivation()),
                   lay.resnetLayer(ms[2], h=0.5, act=act.quadraticActivation()),
                   lay.singleLayer(ms[2], m, act=act.softplusActivation()))

        return f, x

    def test_NN_scalar_output(self):
        torch.set_default_dtype(torch.float64)
        f, x = self.setup_network(1)

        print(self, ': scalar output')
        run_all_tests_laplacian(f, x)

    def test_NN_vector_output(self):
        torch.set_default_dtype(torch.float64)
        m = 3
        f, x = self.setup_network(m)

        print(self, ': vector output')
        run_all_tests_laplacian(f, x)


class TestFullyConnectedNN(unittest.TestCase):

    def test_fullyConnectedNN(self):
        torch.set_default_dtype(torch.float64)
        nex = 11
        d = 3
        widths = [2, 5, 1]

        f = net.fullyConnectedNN([d] + widths, act=act.softplusActivation())

        x = torch.randn(nex, d)

        print(self)
        run_all_tests_laplacian(f, x)


class TestResnetNN(unittest.TestCase):

    def test_resnetNN(self):
        torch.set_default_dtype(torch.float64)
        nex = 11
        width = 3
        depth = 4
        f = net.resnetNN(width, depth, h=0.25, act=act.quadraticActivation())

        x = torch.randn(nex, width)

        print(self)
        run_all_tests_laplacian(f, x)


if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    unittest.main()
