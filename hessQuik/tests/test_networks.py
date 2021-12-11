
import unittest
import torch
import hessQuik.activations as act
import hessQuik.layers as lay
from hessQuik.networks import NN, fullyConnectedNN, resnetNN, ICNN
from hessQuik.tests.utils import run_all_tests


class TestNN(unittest.TestCase):

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

        print(self, ': scalar output')
        run_all_tests(f, x)

    def test_NN_vector_output(self):
        # problem setup
        m = 3
        f, x = self.setup_network(m)

        print(self, ': vector output')
        run_all_tests(f, x)


class TestFullyConnectedNN(unittest.TestCase):

    def test_fullyConnectedNN(self):
        nex = 11
        d = 3
        widths = [2, 7, 5]

        f = fullyConnectedNN([d] + widths, act=act.antiTanhActivation())

        x = torch.randn(nex, d)

        print(self)
        run_all_tests(f, x)


class TestResnetNN(unittest.TestCase):

    def test_resnetNN(self):
        nex = 11
        width = 3
        depth = 4
        f = resnetNN(width, depth, h=0.25, act=act.quadraticActivation())

        x = torch.randn(nex, width)

        print(self)
        run_all_tests(f, x)


class TestICNNNetwork(unittest.TestCase):

    def test_ICNNNetworkLayers(self):
        nex = 11  # no. of examples
        d = 3  # no. of input features
        ms = [None, 5, 2, 7]  # no. of output features

        f = NN(lay.ICNNLayer(d, ms[0], ms[1], act=act.softplusActivation()),
               lay.ICNNLayer(d, ms[1], ms[2], act=act.softplusActivation()),
               lay.ICNNLayer(d, ms[2], ms[3], act=act.softplusActivation()))

        x = torch.randn(nex, d)

        print(self)
        run_all_tests(f, x)

    def test_ICNN(self):
        nex = 11  # no. of examples
        d = 3  # no. of input features
        ms = [None, 5, 2, 7]  # no. of output features

        f = ICNN(d, ms, act=act.softplusActivation())

        x = torch.randn(nex, d)

        print(self)
        run_all_tests(f, x, verbose=True)


class TestBlockNetwork(unittest.TestCase):

    def test_blockFullyConnectedNN(self):
        nex = 11
        d = 3
        widths1 = [2, 3]
        widths2 = [4, 5]
        widths3 = [7, 6, 1]

        f = NN(fullyConnectedNN([d] + widths1, act=act.antiTanhActivation()),
               fullyConnectedNN([widths1[-1]] + widths2, act=act.identityActivation()),
               fullyConnectedNN([widths2[-1]] + widths3, act=act.softplusActivation())
               )

        x = torch.randn(nex, d)

        print(self)
        run_all_tests(f, x)

    def test_blockResnetNN(self):
        nex = 11
        d = 3
        width = 7
        depth = 8

        f = NN(lay.singleLayer(d, width, act=act.antiTanhActivation()),
               resnetNN(width, depth, h=0.8, act=act.softplusActivation()),
               lay.quadraticLayer(width, 6)
               )

        x = torch.randn(nex, d)

        print(self)
        run_all_tests(f, x)

    def test_blockICNN(self):
        nex = 11  # no. of examples
        d = 3  # no. of input features
        m = 4
        ms = [5, 2, 7]  # no. of output features

        f = NN(ICNN(d, [None, m] + ms, act=act.tanhActivation()),
               lay.quadraticICNNLayer(d, ms[-1], 2))

        x = torch.randn(nex, d)

        print(self)
        run_all_tests(f, x)


if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    unittest.main()
