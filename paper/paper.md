---
title: "`hessQuik`: Fast Hessian computation of composite functions"
tags:
  - python
  - pytorch
  - deep neural networks
  - input convex neural networks
authors:
  - name: Elizabeth Newman^[co-first author]
    affiliation: 1
    orcid: 0000-0002-6309-7706
  - name: Lars Ruthotto^[co-first author]
    orcid: 0000-0003-0803-3299
    affiliation: 1
affiliations:
 - name: Emory University, Department of Mathematics
   index: 1
date: 4 February 2022
bibliography: paper.bib
---

\newcommand{\bfK}{\mathbf{K}}
\newcommand{\bfW}{\mathbf{W}}
\newcommand{\bfb}{\mathbf{b}}
\newcommand{\bfu}{\mathbf{u}}
\newcommand{\bfx}{\mathbf{x}}
\newcommand{\bfX}{\mathbf{X}}
\newcommand{\bfe}{\mathbf{e}}
\newcommand{\bfp}{\mathbf{p}}
\newcommand{\bftheta}{\boldsymbol{\theta}}

\newcommand{\R}{\mathbb{R}}
\newcommand{\Rbb}{\mathbb{R}}

\newcommand{\diag}{\text{diag}}


# Summary
`hessQuik` is a lightweight repository for fast computation of second-order derivatives (Hessians) of composite functions (that is, functions formed via compositions) with respect to their inputs.  The core of `hessQuik` is the efficient computation of analytical Hessians with GPU acceleration. `hessQuik` is a PyTorch [@pytorch] package that is user-friendly and easily extendable.  The repository includes a variety of popular functions or layers, including residual layers and input convex layers, from which users can build complex models via composition.  `hessQuik` layers are designed for ease of composition - users need only select the layers and the package provides a convenient wrapper to compose the functions properly.  Each layer provides two modes for derivative computation and the mode is automatically selected to maximize computational efficiency. `hessQuik` includes easy-access, [illustrative tutorials](https://colab.research.google.com/github/elizabethnewman/hessQuik/blob/main/hessQuik/examples/hessQuikPeaksHermiteInterpolation.ipynb) on Google Colaboratory [@googleColab], [reproducible experiments](https://colab.research.google.com/github/elizabethnewman/hessQuik/blob/main/hessQuik/examples/hessQuikTimingTest.ipynb), and unit tests to verify implementations. `hessQuik` enables users to obtain valuable second-order informantion for their models simply and efficiently.

# Statement of need

Deep neural networks (DNNs) and other composition-based models have become a staple of data science, garnering state-of-the-art results in, e.g., image classification and speech recognition [@Goodfellow-et-al-2016], and gaining widespread use in the scientific community, particularly as surrogate models to replace expensive computations [@Anirudh9741]. The unrivaled universality and success of DNNs is due, in part, to the convenience of automatic differentiation (AD) which enables users to compute derivatives of complex functions without an explicit formula. Despite being a powerful tool to compute first-order derivatives (gradients), AD encounters computational obstacles when computing second-order derivatives.  

Knowledge of second-order derivatives is paramount in many growing fields, such as physics-informed neural networks (PINNs) [@Raissi:2019hv], mean-field games [@Ruthotto9183], generative modeling [@ruthotto2021introduction], and adversarial learning [@papernot2016limitations].  In addition, second-order derivatives can provide insight into the optimization problem solved to build a good model [@olearyroseberry2020illposedness]. Hessians are notoriously challenging to compute efficiently with AD and cumbersome to derive and debug analytically.  Hence, many algorithms approximate Hessian information, resulting in suboptimal performance.  To address these challenges, `hessQuik` computes Hessians analytically and efficiently with an implementation that is accelerated on GPUs.


# `hessQuik` Building Blocks

`hessQuik` builds complex functions constructed via composition of simpler functions, which we call \emph{layers}.  The package uses the chain rule to compute Hessians of composite functions, assuming the derivatives of the layers are implemented analytically.  We describe the process mathematically.

Let $f:\Rbb^{n_0} \to \Rbb^{n_\ell}$ be a twice continuously-differentiable function defined as
	\begin{align}
	f = g_{\ell} \circ g_{\ell - 1} \circ \cdots \circ g_1, \quad\text{where} \quad g_i: \Rbb^{n_{i-1}} \to \Rbb^{n_i} \quad \text{for }i=1,\dots, \ell.
	\end{align}
Here, $g_i$ represents the $i$-th layer and $n_i$ is the number of hidden features on the $i$-th layer.   We call $n_0$ the number of input features and $n_{\ell}$ the number of output features. We note that each layer can be parameterized by weights which we can tune by solving an optimization problem. Because `hessQuik` computes derivatives for the network inputs, we omit the weights from our notation. 

## Implemented `hessQuik` Layers

`hessQuik` includes a variety of popular layers and their derivatives.  These layers can be composed to form complex models.  Each layer incorporates a nonlinear activation function, $\sigma: \Rbb \to \Rbb$, that is applied entry-wise.  The `hessQuik` package provides several activation functions, including sigmoid, hyperbolic tangent, and softplus.   Currently supported layers include the following:

* `singleLayer`: This layer consists of an affine transformation followed by pointwise nonlinearity
	\begin{align}
	 g_{\text{single}}(\bfu) = \sigma(\bfK \bfu + \bfb)
	\end{align}
	where $\bfK$ and $\bfb$ are a weight matrix and bias vector, respectively, that can be tuned via optimization methods. Multilayer perceptron neural networks are built upon these layers.
	
* `residualLayer`: This layer differs from a single layer by including a skip connection
	\begin{align}
	g_{\text{residual}}(\bfu) = \bfu + h\sigma(\bfK\bfu + \bfb)
	\end{align}
	where $h > 0$ is a step size.  Residual layers are the building blocks of residual neural networks (ResNets) [@He2016:deep]. ResNets can be interpreted as discretizations of differential equations or dynamical systems [@HaberRuthotto2017; @E2017].
	
* `ICNNLayer`: The input convex neural network layer preserves convexity of the composite function with respect to the input features.  Our layer follows the construction of [@amos2017input].

<!-- 	\begin{align}
	g_{\text{icnn}}(\widetilde{\bfu}) = \sigma(\begin{bmatrix}\bfW^+ & \bfK\end{bmatrix} \widetilde{\bfu} + \bfb), \qquad \widetilde{\bfu} = \begin{bmatrix} \bfu \\ \bfu_0\end{bmatrix}.
	\end{align}
	where $\bfW^+$ has nonnegative entries. -->
	
* `quadraticLayer`, `quadraticICNNLayer`: These are layers that output scalar values and are typically reserved for the final layer of a model.

The variety of implemented layers and activation functions makes designing a wide range of `hessQuik` models easy.


# Computing Derivatives with `hessQuik`

In `hessQuik`, we offer two modes, forward and backward, to compute the gradient $\nabla_{\bfu_0} f$ and the Hessian $\nabla_{\bfu_0}^2 f$ of the function with respect to the input features. The cost of computing derivatives in each mode differs and depends on the number of input and output features.  `hessQuik` automatically selects the least costly method by which to compute derivatives.  We briefly describe the derivative calculations using the two methods.  

First, it is useful to express the evaluation of $f$ as an iterative process.  Let $\bfu_0\in \Rbb^{n_0}$ be a vector of input features.  Then, the function evaluated at $\bfu_0$ is
	\begin{align}
			\bfu_1		&= g_1(\bfu_0)  &&\in \Rbb^{n_1}\\
			\bfu_2		&= g_2(\bfu_1)  &&\in \Rbb^{n_2}\\
						&\vdots \nonumber\\
	f(\bfu_0) \equiv \bfu_{\ell}		&= g_\ell(\bfu_{\ell-1}) &&\in \Rbb^{n_\ell}
	\end{align}
where $\bfu_i$ are the hidden features on layer $i$ for $i=1,\dots,\ell-1$ and $\bfu_{\ell}$ are the output features.

## Forward Mode
Computing derivatives in forward mode means building the gradient and Hessian \emph{during forward propagation}; that is, when we form $\bfu_i$, we simultaneously form the corresponding gradient and Hessian information.  We start by computing the gradient and Hessian of the first layer with respect to the inputs; that is, 
    \begin{align}
        \nabla_{\bfu_0}\bfu_1 &=\nabla_{\bfu_0} g_1(\bfu_0) && \in \Rbb^{n_0\times n_1}\\
        \nabla_{\bfu_0}^2 \bfu_1 &= \nabla_{\bfu_0}^2 g_1(\bfu_0) && \in \Rbb^{n_0\times n_0\times n_1}
    \end{align}
We compute the derivatives of subsequent layers using the following mappings for $i=1, \dots, \ell-1$
    \begin{align}
        \nabla_{\bfu_0}\bfu_{i+1} &= \nabla_{\bfu_0} \bfu_i \nabla_{\bfu_i}g_{i+1}(\bfu_i) && \in \Rbb^{n_0\times n_{i+1}}\\
        \nabla_{\bfu_0}^2\bfu_{i+1} &= \nabla_{\bfu_i}^2g_{i+1}(\bfu_i) \times_1 \nabla_{\bfu_0}\bfu_{i} \times_2 \nabla_{\bfu_0}\bfu_{i}^\top \nonumber\\
	&\qquad + \nabla_{\bfu_0}^2\bfu_i \times_3 \nabla_{\bfu_i}g_{i+1}(\bfu_i) &&\in \Rbb^{n_0\times n_0 \times n_{i+1}}\label{eq:Hessian_ui}
    \end{align}
where $\times_k$ is the mode-$k$ product [@KoldaBader09] and $\nabla_{\bfu_0} \bfu_\ell \equiv \nabla_{\bfu_0} f(\bfu_0)$ is the Hessian we want to compute. 
The Hessian mapping in \autoref{eq:Hessian_ui} is illustrated in \autoref{fig:hessianIllustration}. For efficiency, we store $\nabla_{\bfu_{i}}  g_{i+1}(\bfu_{i})$ when we compute the gradient and re-use this matrix to form the Hessian.  Notice that the sizes of the derivatives always depend on the number of input features, $n_{0}$. 

![Illustration of Hessian computation of $\nabla_{\bfu_0}^ 2\bfu_{i+1}$ in forward mode. Note that for the first term, the gray three-dimensional array $\nabla_{\bfu_i} g_{i+1}(\bfu_i)$ is treated as a stack of matrices.  Then, the same Jacobian matrix $\nabla_{\bfu_0}\bfu_i$ is broadcast to each matrix in the stack, illustrated by the repeated cyan matrices. In the second term, the green matrix $\nabla_{\bfu_i}g_{i+1}(\bfu_i)$ is applied along the third dimension of the magenta three-dimensional array, $\nabla_{\bfu_0}\bfu_i$. Both of these operations can be parallelized and hence accelerated GPUs. \label{fig:hessianIllustration}](img/HessianIllustration.png){ width=100% }

## Backward Mode

Computing derivatives in backward mode is also known as \emph{backward propagation} and is the method by which automatic differentiation computes derivatives.  The process works as follows. We first forward propagate through the network \emph{without computing gradients or Hessians}. After we forward propagate, we build the gradient and Hessian starting from the output layer and working backwards to the input layer. We start by computing derivatives of the final layer with respect to the previous features; that is, 
    \begin{align}
        \nabla_{\bfu_{\ell-1}} \bfu_\ell &= \nabla_{\bfu_{\ell-1}} g_{\ell}(\bfu_{\ell-1}) && \in \Rbb^{n_{\ell-1}\times n_\ell}\\
         \nabla_{\bfu_{\ell-1}}^2 \bfu_\ell &= \nabla_{\bfu_{\ell-1}}^2 g_{\ell}(\bfu_{\ell-1}) && \in \Rbb^{n_{\ell-1}\times n_{\ell-1}\times n_\ell},
    \end{align}
We compute derivatives of previous layers using the following mappings for $i=\ell-1,\dots, 1$:
    \begin{align}
        \nabla_{\bfu_{i-1}} \bfu_{\ell} &= \nabla_{\bfu_{i-1}}  g_{i}(\bfu_{i-1})\nabla_{\bfu_{i}}  \bfu_{\ell} 
			\qquad &&\in \Rbb^{n_{i-1} \times n_{\ell}}\\
		%	
		\nabla_{\bfu_{i-1}}^2 \bfu_{\ell} &= \nabla_{\bfu_i}^2 \bfu_{\ell}  \times_1 \nabla_{\bfu_{i-1}} g_i(\bfu_{i-1}) \times_2 \nabla_{\bfu_{i-1}} g_i(\bfu_{i-1})^\top \nonumber \\
		&\qquad + \nabla_{\bfu_{i-1}}^2 g_i(\bfu_{i-1}) \times_3 \nabla_{\bfu_i} \bfu_{\ell} 
			 \qquad &&\in \Rbb^{n_{i-1} \times n_{i-1} \times n_{\ell}}.
    \end{align}
For efficiency, we re-use $\nabla_{\bfu_{i-1}}  g_{i}(\bfu_{i-1})$ from the gradient computation to compute the Hessian.  Notice that the sizes of the derivatives always depend on the number of output features, $n_{\ell}$. 

## Forward Mode vs. Backward Mode

The computational efficiency of computing derivatives is proportional to the number of input features $n_0$ and the number of output features $n_{\ell}$.  The heuristic we use is if $n_0 < n_\ell$, we compute derivatives in forward mode, otherwise we compute derivatives in backward mode. Our implementation automatically selects the mode of derivative computation based on this heuristic. Users have the option to select their preferred mode of derivative computation if desired. 
	


## Testing Derivative Implementations
The `hessQuik` package includes methods to test derivative implementations and corresponding unit tests.  The main test employs Taylor approximations; for details, see [@haberDerivative]. 


# Efficiency of `hessQuik`

We compare the time to compute the Hessian of a neural network with respect to the input features of three approaches: `hessQuik` (AD-free method), `PytorchAD` which uses automatic differentiation following the implementation in [@huang2021convex], and `PytorchHessian` which uses the built-in Pytorch [Hessian function](https://pytorch.org/docs/stable/generated/torch.autograd\.functional.hessian.html).  

We compare the time to compute the gradient and Hessian of a network with an input dimension $d = 2^k$ where $k=0,1,\dots,10$.  We implement a residual neural network [@He2016:deep] with the width is $w=16$, the depth is $d=4$, and various numbers of output features, $n_\ell$.  For simplicity, the same network architecture is used for every timing test. 

For reproducibility, we compare the time to compute the Hessian using Google Colaboratory (Colab) Pro and provide the notebook in the repository. For CPU runtimes,  Colab Pro uses an Intel(R) Xeon(R) CPU with 2.20GHz processor base speed. For GPU runtimes, Colab Pro uses a Tesla P100 with 16 GB of memory.  

In \autoref{fig:scalar} and \autoref{fig:vector}, we compare the performance of three approaches to compute Hessians of a neural network.  In our experiments, we see faster Hessian computations using `hessQuik` and noticeable acceleration on the GPU, especially for networks with larger input and output dimensions.  Specifically, \autoref{fig:scalar} shows the timing using the `hessQuik` implementation scales better with the number of input features than either of the AD-based methods.  Additionally, \autoref{fig:vector} demonstrates that the `hessQuik` timings remain relatively constant as the number of output features changes whereas the `PytorchAD` timings significantly increase as the number of output features increases.  Note that we only compare to `PytorchAD ` for vector-valued outputs because `PytorchHessian` was noticeably slower for the scalar case.


![Average time over $10$ trials to evaluate and compute the Hessian with respect to the input features for one output feature ($n_{\ell}=1$). Solid lines represent timings on the CPU and dashed lines are timings on the GPU. The circle markers are the timings obtained using `hessQuik`. \label{fig:scalar}](img/hessQuik_timing_scalar.png){ width=80% } 

![Average time over $10$ trials to compute the Hessian with respect to the input features with variable number of input and output features. Each row corresponds to a number of input features, $n_0$, each column corresponds to a number of output features, $n_{\ell}$, and color represents the amount of time to compute (in seconds). \label{fig:vector}](img/hessQuik_timing_vector.png){ width=80% }

# Conclusions
`hessQuik` is a simple, user-friendly repository for computing second-order derivatives of models constructed via composition of functions with respect to their inputs.  This PyTorch package includes many popular built-in layers, tutorial repositories, reproducibile experiments, and unit testing for ease of use.  The implementation scales well in time with the various input and output feature dimensions and performance is accelerated on GPUs, notably faster than automatic-differentiation-based second-order derivative computations.  We hope the accessibility and efficiency of this package will encourage researchers to use and contribute to `hessQuik` in the future.

# Acknowledgements
The development of `hessQuik` was supported in part by the US National Science Foundation under Grant Number 1751636, the Air Force Office of Scientific Research Award FA9550-20-1-0372, and the US DOE Office of Advanced Scientific Computing Research Field Work Proposal 20-023231. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the funding agencies.

# References
