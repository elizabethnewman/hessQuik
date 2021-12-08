import torch
import math
from layer_types import hessQuikLayer, activationFunction
from hessQuik import activation_functions as act


class singleLayer(hessQuikLayer):
    """
    Forward propagation through single layer of the form

        f(u(x)) = act(u(x) @ K + b).

    Here, u(x) is the input into the layer and x is the input into the network of shapes

        x : (N, d) torch.Tensor
        u(x) : (N, in_features) torch.Tensor
        f(u(x)) : (N, out_features) torch.Tensor

    where N is the number of examples and d is the number of input features into the network.
    """

    def __init__(self, in_features, out_features, act: activationFunction = act.identityActivation(),
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(singleLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.act = act

        self.K = nn.Parameter(torch.empty(in_features, out_features, **factory_kwargs))
        self.b = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.K, a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.K)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.b, -bound, bound)

    def forward(self, u, do_gradient=False, do_Hessian=False, dudx=None, d2ud2x=None, reverse_mode=False):
        """
        Forward propagate through singleLayer

        Parameters
        ----------
        u : (N, in_features) torch.Tensor
            Tensor of input features
        do_gradient : bool, optional
            If True, compute the gradient of the output of the layer, f(u(x)), with respect to the network input, x
        do_Hessian : bool, optional
            If True, compute the Hessian of the output of the layer, f(u(x)), with respect to the network input, x
        dudx : (N, d, in_features) torch.Tensor
            Gradient of the input into the layer, u(x), with respect to the network inputs, x
        d2ud2x : (N, d, d, in_features) torch.Tensor
            Hessian of the input into the layer, u(x), with respect to the network inputs, x

        Returns
        -------
        f : (N, out_features) torch.Tensor
            Output of the layer, f(u(x))
        dfdx : (N, d, out_features)
            Gradient of the output of the layer, f(u(x)), with respect to the network inputs, x
        d2fd2x : (N, d, d, out_features)
            Hessian of the output of the layer, f(u(x)), with respect to the network inputs, x
        """
        (dfdx, d2fd2x) = (None, None)
        f, dsig, d2sig = self.act.forward(u @ self.K + self.b, do_gradient=do_gradient, do_Hessian=do_Hessian,
                                          reverse_mode=reverse_mode)

        if (do_gradient or do_Hessian) and not reverse_mode:
            dfdx = dsig.unsqueeze(1) * self.K

            # -------------------------------------------------------------------------------------------------------- #
            if do_Hessian:
                d2fd2x = (d2sig.unsqueeze(1) * self.K).unsqueeze(2) * self.K.unsqueeze(0).unsqueeze(0)

                # Gauss-Newton approximation
                if dudx is not None:
                    d2fd2x = dudx.unsqueeze(1) @ (d2fd2x.permute(0, 3, 1, 2) @ dudx.unsqueeze(1).permute(0, 1, 3, 2))
                    d2fd2x = d2fd2x.permute(0, 2, 3, 1)

                if d2ud2x is not None:
                    # extra term to compute full Hessian
                    d2fd2x += d2ud2x @ dfdx.unsqueeze(1)  # I already compute this in gradient
            # -------------------------------------------------------------------------------------------------------- #
            # finish computing gradient
            if dudx is not None:
                dfdx = dudx @ dfdx

        return f, dfdx, d2fd2x

    def backward(self, do_Hessian=False, dgdf=None, d2gd2f=None):
        """
        Backward propagate through singleLayer

        Parameters
        ----------
        do_Hessian : bool, optional
            If True, compute the Hessian of the output of the layer, g(f(x)), with respect to the network input, x
        dgdf : (N, d, out_features) torch.Tensor
            Gradient of the input into the layer, g(f(x)), with respect to the layer outputs, f(x)
        d2gd2f : (N, d, d, out_features) torch.Tensor
            Hessian of the input into the layer, g(f(x)), with respect to the layer outputs, f(x)

        Returns
        -------
        dgdx : (N, d, in_features)
            Gradient of the output of the layer, g(f(x)), with respect to the network inputs, x
        d2gd2x : (N, d, d, in_features)
            Hessian of the output of the layer, g(f(x)), with respect to the network inputs, x
        """
        d2gd2x = None
        dsig, d2sig = self.act.backward(do_Hessian=do_Hessian)
        dgdx = dsig.unsqueeze(1) * self.K

        if do_Hessian:
            # TODO: change order of operations, multiply K's first; check if logic with better naming
            d2gd2x = (d2sig.unsqueeze(1) * self.K.unsqueeze(0)).unsqueeze(2) * self.K.unsqueeze(0).unsqueeze(0)
            if d2gd2f is not None:
                # Gauss-Newton approximation
                h1 = (dgdx.unsqueeze(1) @ d2gd2f.permute(0, 3, 1, 2) @ dgdx.permute(0, 2, 1).unsqueeze(1))
                h1 = h1.permute(0, 2, 3, 1)

                # extra term to compute full Hessian
                N, _, _, m = d2gd2x.shape
                h2 = d2gd2x.view(N, -1, m) @ dgdf.view(N, m, -1)
                h2 = h2.view(h1.shape)

                # combine
                d2gd2x = h1 + h2

        # finish computing gradient
        if dgdf is not None:
            dgdx = dgdx @ dgdf

        return dgdx, d2gd2x

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
