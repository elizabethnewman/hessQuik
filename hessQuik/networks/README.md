
### Basics
Every hessQuik network is constructed using the ```NN``` wrapper.  For example,
```python
layer = lay.singleLayer(3, 4)
f = net.NN(layer)

nex = 100
x = torch.randn(nex, net.dim_input())
f0, df0, d2f0 = f(x, do_gradient=True, do_Hessian=True)
```

Unless specified by the user, forward and backward mode is automatically selected based on the dimensions of the input and output features.  If ```net.dim_input() < net.dim_output()```, then derviatives are computed in forward mode. 

### Options
For convenience, we provide some commonly-used network architectures
```python
fcnet = net.fullyConnectedNN(widths=[32, 64, 20], act=act.softplusActivation())
resnet = net.resnetNN(width=32, depth=8, h=0.25, act=act.antiTanhActivation())
icnn = net.ICNN(input_dim=10, widths=[32, 64, 20], act=act.softplusActivation())
```

You can create a hessQuik network using layers and these commonly-used architectures as blocks via
```python
d = 10 # number of input features
width = 32
f = net.NN(lay.singleLayer(d, width, act=act.softplusActivation()), 
             net.resnetNN(width=width, depth=8, h=0.25, act=act.antiTanhActivation()), 
             lay.quadraticLayer(width, rank=5))

icnn = net.NN(net.ICNN(input_dim=10, widths=[32, 64, 20], act=act.antiTanhActivation()), 
              lay.quadraticICNNLayer(input_dim=10, in_features=20, rank=2))
```
We do not allow concatenation of ICNN layers and networks with non-ICNN layers.

We offer some additional wrappers for comparison
```python
fAD = net.NNPytorchAD(f)         # use PyTorch automatic differentiation
fHess = net.NNPytorchHessian(f)  # use PyTorch hessian function
```
The ```NNPytorchAD``` wrapper using the implementation from [CP-Flow](https://github.com/CW-Huang/CP-Flow).  Currently, the ```NNPytorchHessian``` wrapper is only available for networks with scalar outputs.

### For Contributors
To create new networks, you should inherit the attributes and methods from ```NN```.  You should not need to rewrite the ```NN``` methods, only change the list of layers and networks as inputs.

To test new networks, add the appropriate tests in ```tests/test_networks.py```.