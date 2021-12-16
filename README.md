# hessQuik

## Installation

### For Users: Install with **pip**

```console
python -m pip install git+https://github.com/elizabethnewman/hessQuik.git
```

[comment]: <> (virtualenv -p python env_name)

[comment]: <> (source env_name/bin/activate)

### For Contributors: Clone with **git**

```console
git clone https://github.com/elizabethnewman/hessQuik.git
```



[comment]: <> (Install package)

[comment]: <> ([comment]: <> &#40;https://adamj.eu/tech/2019/03/11/pip-install-from-a-git-repository/&#41;)

[comment]: <> (```html)

[comment]: <> (python -m pip install git+https://github.com/elizabethnewman/hessQuik.git)

[comment]: <> (```)

[comment]: <> (If the repository is private, use)

[comment]: <> (```html)

[comment]: <> (python -m pip install git+ssh://git@github.com/elizabethnewman/hessQuik.git)

[comment]: <> (```)

[comment]: <> (Make sure to import torch before importing hessQuik &#40;this is a bug currently&#41;)

[comment]: <> (If hessQuik updated, reinstall via one of the following:)

[comment]: <> (```html)

[comment]: <> (pip install --upgrade --force-reinstall <package>)

[comment]: <> (pip install -I <package>)

[comment]: <> (pip install --ignore-installed <package>)

[comment]: <> (```)

[comment]: <> (When finished, deactivate virtual environment.)

[comment]: <> (```html)

[comment]: <> (deactivate)

[comment]: <> (```)

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

You can obtain gradients and Hessians via
```python
nex = 20 # number of examples
x = torch.randn(nex, d)
fx, dfx, d2fx = f(x, do_gradient=True, do_Hessian=True)
```

[comment]: <> (or in backward mode via)

[comment]: <> (```python)

[comment]: <> (nex = 20 # number of examples)

[comment]: <> (x = torch.randn&#40;nex, d&#41;)

[comment]: <> (fx, *_ = f&#40;x, reverse_mode=True&#41;)

[comment]: <> (dfx, d2fx = f.backward&#40;do_Hessian=True&#41;)

[comment]: <> (```)


## Examples
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1GCUSR9fGhQ9PoqfPxv8qRfqf88_ibyUA?usp=sharing) Practical Use: Hermite Interpolation

[comment]: <> ([![Open In Colab]&#40;https://colab.research.google.com/assets/colab-badge.svg&#41;]&#40;https://colab.research.google.com/drive/1zTgU0pcZJMRmSL4Rgt_oNSYcBI2cIj04?usp=sharing&#41; Timing Test)

[comment]: <> ([![Open In Colab]&#40;https://colab.research.google.com/assets/colab-badge.svg&#41;]&#40;https://colab.research.google.com/drive/1C-CQbOSGuSkXbpfLo2zlP2BQJJwegI09?usp=sharing&#41; hessQuik Profiler)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1842TWdILPNhiqLMw9JMZjdZ6T-B6hzul?usp=sharing) Tutorial: Constructing and Testing ```hessQuik``` Layers

