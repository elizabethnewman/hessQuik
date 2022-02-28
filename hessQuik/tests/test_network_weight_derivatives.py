import unittest
import torch
import hessQuik.activations as act
import hessQuik.layers as lay
import hessQuik.networks as net
from hessQuik.utils import network_derivative_check


class TestNetworkWeightDerivatives(unittest.TestCase):

    @staticmethod
    def setup_data():
        torch.set_default_dtype(torch.float64)
        nex = 11  # no. of examples
        d = 4  # no. of input features
        x = torch.randn(nex, d)
        return x

    def test_singleLayer(self):
        torch.set_default_dtype(torch.float64)
        x = self.setup_data()
        d = x.shape[1]
        f = lay.singleLayer(d, 7, act=act.softplusActivation())

        print(self)
        network_derivative_check(f, x, do_Hessian=True, forward_mode=True)
        network_derivative_check(f, x, do_Hessian=True, forward_mode=False)

    def test_resnetNN(self):
        torch.set_default_dtype(torch.float64)
        x = self.setup_data()
        d = x.shape[1]
        f = net.resnetNN(d, 4, act=act.softplusActivation())

        print(self)
        network_derivative_check(f, x, do_Hessian=True, forward_mode=True)
        network_derivative_check(f, x, do_Hessian=True, forward_mode=False)

    def test_blockNN(self):
        torch.set_default_dtype(torch.float64)
        x = self.setup_data()
        d = x.shape[1]
        width = 7
        f = net.NN(lay.singleLayer(d, width, act=act.tanhActivation()),
                   net.resnetNN(width, 4, act=act.softplusActivation()),
                   net.fullyConnectedNN([width, 13, 5], act=act.quadraticActivation()),
                   lay.singleLayer(5, 3, act=act.identityActivation()),
                   lay.quadraticLayer(3, 2)
                   )

        print(self)
        network_derivative_check(f, x, do_Hessian=True, forward_mode=True)
        network_derivative_check(f, x, do_Hessian=True, forward_mode=False)

    def test_ICNN(self):
        torch.set_default_dtype(torch.float64)
        x = self.setup_data()
        d = x.shape[1]
        m = 4
        ms = [5, 2, 7]  # no. of output features

        f = net.NN(net.ICNN(d, [None, m] + ms, act=act.tanhActivation()),
                   lay.quadraticICNNLayer(d, ms[-1], 2))

        print(self)
        network_derivative_check(f, x, do_Hessian=True, forward_mode=True)
        network_derivative_check(f, x, do_Hessian=True, forward_mode=False)


if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    unittest.main()
