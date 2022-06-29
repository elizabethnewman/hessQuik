# hessQuik
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04171/status.svg)](https://doi.org/10.21105/joss.04171)

A lightweight package for fast, GPU-accelerated computation of gradients and Hessians of functions constructed via composition.

## Statement of Need
Deep neural networks (DNNs) and other composition-based models have become a staple of data science, garnering state-of-the-art results and gaining widespread use in the scientific community, particularly as surrogate models to replace expensive computations. The unrivaled universality and success of DNNs is due, in part, to the convenience of automatic differentiation (AD) which enables users to compute derivatives of complex functions without an explicit formula. Despite being a powerful tool to compute first-order derivatives (gradients), AD encounters computational obstacles when computing second-order derivatives (Hessians).  

Knowledge of second-order derivatives is paramount in many growing fields and can provide insight into the optimization problem solved to build a good model. Hessians are notoriously challenging to compute efficiently with AD and cumbersome to derive and debug analytically.  Hence, many algorithms approximate Hessian information, resulting in suboptimal performance.  To address these challenges, `hessQuik` computes Hessians analytically and efficiently with an implementation that is accelerated on GPUs.

## Documentation

For package usage and details, see our paper in the [Journal of Open Source Software](https://joss.theoj.org/papers/10.21105/joss.04171#).

For detailed documentation, visit [https://hessquik.readthedocs.io/](https://hessquik.readthedocs.io/en/latest/index.html).

## Installation

From PyPI:
```console
pip install hessQuik
```

From Github:
```console
python -m pip install git+https://github.com/elizabethnewman/hessQuik.git
```

### Dependencies
These dependencies are installed automatically with ```pip```.
* torch (recommended version >= 1.10.0, but code will run with version >= 1.5.0)

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

### Support for Laplacians and Directional Derivatives

If you only require Laplacians, not full Hessians, you can obtain the gradients and Laplacians via
```python
fx, dfx, lapfd2x = f(x, do_gradient=True, do_Laplacian=True)
```

If you only require evaluations of the Jacobian and Hessian along certain directions, you can provide the directions in `forward_mode` via
```python
k = 3  # number of directions
v = torch.randn(k, d)
fx, vdfx, vd2fxv = f(x, do_gradient=True, do_Hessian=True, v=v, forward_mode=True)
```
and in `backward_mode` via
```python
m = widths[-1]  # dimension of output features
v = torch.randn(m, k)
fx, dfxv, d2fxv = f(x, do_gradient=True, do_Hessian=True, v=v, forward_mode=False)
```

**Currently, this functionality is only supported for `singleLayer`, `resnetLayer`, and networks using only these types of layers, including `fullyConnectedNN` and `resnetNN`.**


## Examples
To make the code accessible, we provide some introductory Google Colaboratory notebooks.

[Practical Use: Hermite Interpolation](https://github.com/elizabethnewman/hessQuik/blob/main/hessQuik/examples/hessQuikPeaksHermiteInterpolation.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elizabethnewman/hessQuik/blob/main/hessQuik/examples/hessQuikPeaksHermiteInterpolation.ipynb) 

[Tutorial: Constructing and Testing ```hessQuik``` Layers](https://github.com/elizabethnewman/hessQuik/blob/main/hessQuik/examples/hessQuikSingleLayerTutorial.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elizabethnewman/hessQuik/blob/main/hessQuik/examples/hessQuikSingleLayerTutorial.ipynb)

[Timing Comparisons](https://github.com/elizabethnewman/hessQuik/blob/main/hessQuik/examples/hessQuikTimingTest.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elizabethnewman/hessQuik/blob/main/hessQuik/examples/hessQuikTimingTest.ipynb)

## Contributing

To contribute to ```hessQuik```, follow these steps:
1. [Fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) the ```hessQuik``` repository
2. Clone your fork using 
```console
git clone https://github.com/<username>/hessQuik.git
```
3. Contribute to your forked repository
4. Create a [pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request)

If your code passes the necessary numerical tests and is well-documented, your changes and/or additions will be merged in the main ```hessQuik``` repository. You can find examples of the tests used in each file and related unit tests the ```tests``` directory.

## Reporting Bugs

If you notice an issue with this repository, please report it using [Github Issues](https://docs.github.com/en/issues/tracking-your-work-with-issues/about-issues).  When reporting an implementation bug, include a small example that helps to reproduce the error.  The issue will be addressed as quickly as possible.

## How to Cite

```
@article{Newman2022,
  doi = {10.21105/joss.04171},
  url = {https://doi.org/10.21105/joss.04171},
  year = {2022},
  publisher = {The Open Journal},
  volume = {7},
  number = {72},
  pages = {4171},
  author = {Elizabeth Newman and Lars Ruthotto},
  title = {`hessQuik`: Fast Hessian computation of composite functions},
  journal = {Journal of Open Source Software}
}
```


## Acknowledgements

This material is in part based upon work supported by the US National Science Foundation under Grant Number 1751636, the Air Force Office of Scientific Research Award FA9550-20-1-0372, and the US DOE Office of
Advanced Scientific Computing Research Field Work Proposal 20-023231. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the funding agencies.
