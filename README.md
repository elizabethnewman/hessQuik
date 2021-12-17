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


## Examples
To make the code accessible, we provide some Google Colaboratory notebooks.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1GCUSR9fGhQ9PoqfPxv8qRfqf88_ibyUA?usp=sharing) Practical Use: Hermite Interpolation

[comment]: <> ([![Open In Colab]&#40;https://colab.research.google.com/assets/colab-badge.svg&#41;]&#40;https://colab.research.google.com/drive/1zTgU0pcZJMRmSL4Rgt_oNSYcBI2cIj04?usp=sharing&#41; Timing Test)

[comment]: <> ([![Open In Colab]&#40;https://colab.research.google.com/assets/colab-badge.svg&#41;]&#40;https://colab.research.google.com/drive/1C-CQbOSGuSkXbpfLo2zlP2BQJJwegI09?usp=sharing&#41; hessQuik Profiler)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1842TWdILPNhiqLMw9JMZjdZ6T-B6hzul?usp=sharing) Tutorial: Constructing and Testing ```hessQuik``` Layers

