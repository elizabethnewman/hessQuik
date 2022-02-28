Getting Started
===============

Once ``hessQuik`` is installed, you can import as follows::

    import hessQuik.activations as act
    import hessQuik.layers as lay
    import hessQuik.networks as net

You can construct a ``hessQuik`` network from layers as follows::

    d = 10 # dimension of the input features
    widths = [32, 64] # hidden channel dimensions
    f = net.NN(lay.singleLayer(d, widths[0], act.antiTanhActivation()),
        lay.resnetLayer(widths[0], h=1.0, act.softplusActivation()),
        lay.singleLayer(widths[0], widths[1], act.quadraticActivation())
        )

You can obtain gradients and Hessians via::

    nex = 20 # number of examples
    x = torch.randn(nex, d)
    fx, dfx, d2fx = f(x, do_gradient=True, do_Hessian=True)

That's it!  You now have computed the value, gradient, and Hessian of the network :math:`f` at the point :math:`x`.