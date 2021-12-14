
### Usage
The hessQuik network wrappers are located in **hessQuik_network.py** and are used as follows:
```python
layer = lay.singleLayer(3, 4)
net1 = net.NN(layer)
net2 = net.NNPytorchAD(net1)
```

Here, **net1** uses our hessQuik implementation to compute gradients and Hessians and **net2** uses the implementations found in [CP-Flow](https://github.com/CW-Huang/CP-Flow). 


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
net = net.NN(lay.singleLayer(d, width, act=act.softplusActivation()), 
             net.resnetNN(width=width, depth=8, h=0.25, act=act.antiTanhActivation()), 
             lay.quadraticLayer(width, rank=5))

icnn = net.NN(net.ICNN(input_dim=10, widths=[32, 64, 20], act=act.antiTanhActivation()), 
              lay.quadraticICNNLayer(input_dim=10, in_features=20, rank=2))
```
We do not allow concatenation of ICNN layers and networks with non-ICNN layers. 