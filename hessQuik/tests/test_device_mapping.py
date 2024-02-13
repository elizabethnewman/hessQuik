import unittest
import torch
import hessQuik.activations as act
import hessQuik.layers as lay
import hessQuik.networks as net


class TestLayer(unittest.TestCase):

    def test_singleLayer(self):
        torch.set_default_dtype(torch.float32)
        device = 'mps'
        nex = 11  # no. of examples
        d = 4  # no. of input features
        m = 7  # no. of output features
        x = torch.randn(nex, d, device=device)
        f = lay.singleLayer(d, m, act=act.softplusActivation()).to(device)

        print(self)
        f(x, do_gradient=True, do_Hessian=True, forward_mode=True)
        f(x, do_gradient=True, do_Hessian=True, forward_mode=False)

    def test_resnetLayer(self):
        torch.set_default_dtype(torch.float32)
        device = 'mps'

        nex = 11  # no. of examples
        width = 4  # no. of input features
        h = 0.25
        x = torch.randn(nex, width, device=device)
        f = lay.resnetLayer(width, h=h, act=act.softplusActivation()).to(device)
        print(self)
        f(x, do_gradient=True, do_Hessian=True, forward_mode=True)
        f(x, do_gradient=True, do_Hessian=True, forward_mode=False)

    def test_ICNNLayer(self):
        torch.set_default_dtype(torch.float32)
        device = 'mps'

        nex = 11  # no. of examples
        d = 3  # no. of input features
        m = 5  # no. of output features
        x = torch.randn(nex, d, device=device)
        f = lay.ICNNLayer(d, None, m, act=act.softplusActivation()).to(device)

        print(self)
        f(x, do_gradient=True, do_Hessian=True, forward_mode=True)
        f(x, do_gradient=True, do_Hessian=True, forward_mode=False)

    def test_quadraticLayer(self):
        torch.set_default_dtype(torch.float32)
        device = 'mps'

        nex = 11  # no. of examples
        d = 4  # no. of input dimensiona features
        m = 7  # rank
        x = torch.randn(nex, d).to(device)
        f = lay.quadraticLayer(d, m).to(device)

        print(self)
        f(x, do_gradient=True, do_Hessian=True, forward_mode=True)
        f(x, do_gradient=True, do_Hessian=True, forward_mode=False)

    def test_quadraticICNNLayer(self):
        torch.set_default_dtype(torch.float32)
        device = 'mps'

        nex = 11  # no. of examples
        d = 3  # no. of input features
        m = 5  # no. of output features

        x = torch.randn(nex, d).to(device)
        f = lay.quadraticICNNLayer(d, None, m).to(device)

        print(self)
        f(x, do_gradient=True, do_Hessian=True, forward_mode=True)
        f(x, do_gradient=True, do_Hessian=True, forward_mode=False)


class TestNN(unittest.TestCase):

    @staticmethod
    def setup_network(m):
        torch.set_default_dtype(torch.float32)
        device = 'mps'

        nex = 11
        d = 3
        ms = [2, 7, 5]
        x = torch.randn(nex, d).to(device)

        f = net.NN(lay.singleLayer(d, ms[0], act=act.softplusActivation()),
                   lay.singleLayer(ms[0], ms[1], act=act.softplusActivation()),
                   lay.resnetLayer(ms[1], h=0.25, act=act.antiTanhActivation()),
                   lay.singleLayer(ms[1], ms[2], act=act.softplusActivation()),
                   lay.resnetLayer(ms[2], h=0.5, act=act.quadraticActivation()),
                   lay.singleLayer(ms[2], m, act=act.softplusActivation())).to(device)

        return f, x

    def test_NN_scalar_output(self):
        torch.set_default_dtype(torch.float32)
        f, x = self.setup_network(1)

        print(self, ': scalar output')
        f(x, do_gradient=True, do_Hessian=True, forward_mode=True)
        f(x, do_gradient=True, do_Hessian=True, forward_mode=False)

    def test_NN_vector_output(self):
        torch.set_default_dtype(torch.float32)
        m = 3
        f, x = self.setup_network(m)

        print(self)
        f(x, do_gradient=True, do_Hessian=True, forward_mode=True)
        f(x, do_gradient=True, do_Hessian=True, forward_mode=False)


class TestFullyConnectedNN(unittest.TestCase):

    def test_fullyConnectedNN(self):
        torch.set_default_dtype(torch.float32)
        device = 'mps'

        nex = 11
        d = 3
        widths = [2, 5, 1]

        f = net.fullyConnectedNN([d] + widths, act=act.softplusActivation()).to(device)

        x = torch.randn(nex, d).to(device)

        print(self)
        f(x, do_gradient=True, do_Hessian=True, forward_mode=True)
        f(x, do_gradient=True, do_Hessian=True, forward_mode=False)


class TestResnetNN(unittest.TestCase):

    def test_resnetNN(self):
        torch.set_default_dtype(torch.float32)
        device = 'mps'

        nex = 11
        width = 3
        depth = 4
        f = net.resnetNN(width, depth, h=0.25, act=act.quadraticActivation()).to(device)

        x = torch.randn(nex, width).to(device)

        print(self)
        f(x, do_gradient=True, do_Hessian=True, forward_mode=True)
        f(x, do_gradient=True, do_Hessian=True, forward_mode=False)


class TestICNNNetwork(unittest.TestCase):

    @staticmethod
    def setup_data():
        torch.set_default_dtype(torch.float32)
        device = 'mps'

        nex = 11
        d = 3
        x = torch.randn(nex, d).to(device)
        return x

    def test_ICNNNetworkLayers(self):
        torch.set_default_dtype(torch.float32)
        device = 'mps'

        x = self.setup_data().to(device)
        nex = 11  # no. of examples
        d = x.shape[1]  # no. of input features
        ms = [None, 5, 2, 7]  # no. of output features

        f = net.NN(lay.ICNNLayer(d, ms[0], ms[1], act=act.softplusActivation()),
                   lay.ICNNLayer(d, ms[1], ms[2], act=act.softplusActivation()),
                   lay.ICNNLayer(d, ms[2], ms[3], act=act.softplusActivation())).to(device)

        print(self)
        f(x, do_gradient=True, do_Hessian=True, forward_mode=True)
        f(x, do_gradient=True, do_Hessian=True, forward_mode=False)

    def test_ICNN(self):
        torch.set_default_dtype(torch.float32)
        device = 'mps'

        x = self.setup_data().to(device)
        d = x.shape[1]
        ms = [None, 5, 2, 7]  # no. of output features

        f = net.ICNN(d, ms, act=act.softplusActivation()).to(device)

        print(self)
        f(x, do_gradient=True, do_Hessian=True, forward_mode=True)
        f(x, do_gradient=True, do_Hessian=True, forward_mode=False)


class TestBlockNetwork(unittest.TestCase):

    @staticmethod
    def setup_data():
        torch.set_default_dtype(torch.float32)
        device = 'mps'

        nex = 11
        d = 3
        x = torch.randn(nex, d).to(device)
        return x

    def test_blockFullyConnectedNN(self):
        torch.set_default_dtype(torch.float32)
        device = 'mps'

        x = self.setup_data().to(device)
        d = x.shape[1]
        widths1 = [2, 3]
        widths2 = [4, 5]
        widths3 = [7, 6, 1]

        f = net.NN(net.fullyConnectedNN([d] + widths1, act=act.antiTanhActivation()),
                   net.fullyConnectedNN([widths1[-1]] + widths2, act=act.identityActivation()),
                   net.fullyConnectedNN([widths2[-1]] + widths3, act=act.softplusActivation())
                   ).to(device)

        print(self)
        f(x, do_gradient=True, do_Hessian=True, forward_mode=True)
        f(x, do_gradient=True, do_Hessian=True, forward_mode=False)

    def test_blockResnetNN(self):
        torch.set_default_dtype(torch.float32)
        device = 'mps'

        x = self.setup_data().to(device)
        d = x.shape[1]
        width = 7
        depth = 3

        f = net.NN(lay.singleLayer(d, width, act=act.antiTanhActivation()),
                   net.resnetNN(width, depth, h=0.8, act=act.softplusActivation()),
                   lay.quadraticLayer(width, 2)
                   ).to(device)

        print(self)
        f(x, do_gradient=True, do_Hessian=True, forward_mode=True)
        f(x, do_gradient=True, do_Hessian=True, forward_mode=False)

    def test_blockICNN(self):
        torch.set_default_dtype(torch.float32)
        device = 'mps'
        x = self.setup_data().to(device)
        d = x.shape[1]
        m = 4
        ms = [5, 2, 7]  # no. of output features

        f = net.NN(net.ICNN(d, [None, m] + ms, act=act.tanhActivation()),
                   lay.quadraticICNNLayer(d, ms[-1], 2)).to(device)

        print(self)
        f(x, do_gradient=True, do_Hessian=True, forward_mode=True)
        f(x, do_gradient=True, do_Hessian=True, forward_mode=False)




if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    unittest.main()
