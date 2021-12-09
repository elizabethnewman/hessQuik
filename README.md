# hessQuik


## Installation

Create virtual environment

```html
virtualenv -p python env_name
source env_name/bin/activate
```


Install package

[comment]: <> (https://adamj.eu/tech/2019/03/11/pip-install-from-a-git-repository/)
```html
python -m pip install git+https://github.com/elizabethnewman/hessQuik.git
```
If the repository is private, use
```html
python -m pip install git+ssh://git@github.com/elizabethnewman/hessQuik.git
```

[comment]: <> (Make sure to import torch before importing hessQuik &#40;this is a bug currently&#41;)

If hessQuik updated, reinstall via one of the following:
```html
pip install --upgrade --force-reinstall <package>
pip install -I <package>
pip install --ignore-installed <package>
```

When finished, deactivate virtual environment.

```html
deactivate
```

## Getting Started

Once you have installed hessQuik, you can import as follows:
```python
import hessQuik.activations as act
import hessQuik.layers as lay
import hessQuik.networks as net
```

You can construct a hessQuik network from layers as follows:
```python
d = 10 # dimension of the input features
widths = [32, 64] # hidden channel dimensions
f = net.NN(lay.singleLayer(d, widths[0], act.antiTanhActivation()), 
           lay.resnetLayer(widths[0], h=1.0, act.softplusActivation()),
           lay.singleLayer(widths[0], widths[1], act.quadraticActivation())
           )
```

Once you have constructed the network, you can run forward propagation and obtain the gradient and Hessian as follows:
```python
nex = 20 # number of examples
x = torch.randn(nex, d)
fx, dfx, d2fx = f(x, do_gradient=True, do_Hessian=True)
```

## Examples
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1GCUSR9fGhQ9PoqfPxv8qRfqf88_ibyUA?usp=sharing) Peaks Hermite Interpolation