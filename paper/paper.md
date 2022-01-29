---
title: 'hessQuik: Fast gradient and Hessian computations of deep neural networks'
tags:
  - Python
  - Pytorch
  - deep neural networks
  - input convex neural networks
authors:
  - name: Lars Ruthotto^[co-first author]
    orcid: 0000-0003-0803-3299
    affiliation: 1
  - name: Elizabeth Newman^[co-first author]
    affiliation: 1
    orcid: 0000-0002-6309-7706
affiliations:
 - name: Emory University
   index: 1
date: 29 January 2022
bibliography: paper.bib
---

\newcommand{\bfK}{\mathbf{K}}
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
`hessQuik` provides efficient methods of computing second-order derivatives (Hessians) with respect to the inputs of (parameterized) functions formed via composition, such as neural networks, without automatic differentiation (AD). In general, Hessians are challenging to compute efficiently with AD and cumbersome to derive and debug analytically.  Hence, many algorithms approximate Hessian information, resulting in suboptimal performance.  To address these challenges of second-order derivative computation, `hessQuik` provides efficient, analytic Hessians with GPU acceleration for a variety of network layers that can be used seamlessly with PyTorch. The package is user-friendly and flexible by design -- users can construct many-layered networks with a variety of layer types, including fully-connected layers, residual layers, and input convex layers, and use various nonlinear activation functions, including hyperbolic tangent, sigmoid, and quadratic.  


# Statement of need

Deep neural networks (DNNs), and other composition-based functions, have become a staple of data science, garnering state-of-the-art results in, e.g., image classification and speech recognition, and gaining widespread use in the scientific community, particularly as surrogate models to replace expensive computations. 
The unrivaled universality and success of DNNs is due, in part, to the convenience of automatic differentiation (AD) which enables users to compute derivatives of complex functions without an explicit formula. 
Despite being a powerful tool to compute gradients, AD encounters computational obstacles when computing second-order derivatives. Knowledge of second-order derivatives is paramount in many growing fields, such as physics-informed neural networks (PINNs) [@Raissi:2019hv], mean-field games [@Ruthotto9183], generative modeling [@ruthotto2021introduction], and adversarial learning [@papernot2016limitations].  In addition, second-order derivative information can provide insight into the optimization problem solved to form a good DNN model [@olearyroseberry2020illposedness]. `hessQuik` addresses these challenges of computing second-order derivatives 

# The Chain Rule for Composed Functions

We consider complex functions constructed via composition of simpler functions, which we call \emph{layers}.  Specifically, let $f:\Rbb^{n_0} \to \Rbb^{n_\ell}$ be a twice continuously-differentiable function defined as
	\begin{align}
	f = g_{\ell} \circ g_{\ell - 1} \circ \cdots \circ g_1, \quad\text{where} \quad g_i: \Rbb^{n_{i-1}} \to \Rbb^{n_i} \quad \text{for }i=1,\dots, \ell.
	\end{align}
Here, $g_i$ represents the $i$-th layer and $n_i$ is the number of hidden features on the $i$-th layer.   We call $n_0$ the number of input features and $n_{\ell}$ the number of output features.  `hessQuik` provides many common layer types, including fully-connected and residual layers.  We note that each layer is parameterized by weights which we can tune by solving an optimization problem. Because `hessQuik` computes derivatives for the network inputs for a given set of weights, we omit the network weights from our notation. 

It is useful to express the evaluation of $f$ as an iterative process.  Let $\bfu_0\in \Rbb^{n_0}$ be a vector of input features.  Then, the function evaluated at $\bfu_0$ is
	\begin{align}
			\bfu_1		&= g_1(\bfu_0)  &&\in \Rbb^{n_1}\\
			\bfu_2		&= g_2(\bfu_1)  &&\in \Rbb^{n_2}\\
						&\vdots \nonumber\\
	f(\bfu_0) \equiv \bfu_{\ell}		&= g_\ell(\bfu_{\ell-1}) &&\in \Rbb^{n_\ell}
	\end{align}
where $\bfu_i$ are the hidden features on layer $i$ for $i=1,\dots,\ell-1$ and $\bfu_{\ell}$ are the output features.

In `hessQuik`, we offer two modes, forward and backward, to compute the gradient $\nabla_{\bfu_0} f$ and the Hessian $\nabla_{\bfu_0}^2 f$ of the function with respect to the input features. The cost of computing derivatives in each mode is different and depends on the number of input and output features.  `hessQuik` automatically selects the least costly mode through which to compute derivatives.  We briefly describe the derivative calculations using the two methods.  

## Computing Derivatives in Forward Mode
Computing derivatives in forward mode means we sequentially build the gradient and Hessian \emph{during forward propagation}; that is, when we form $\bfu_i$, we simultaneously form the corresponding gradient and Hessian information.  We start by computing the gradient and Hessian of the first layer with respect to the inputs; that is, 
    \begin{align}
        \nabla_{\bfu_0}\bfu_1 &=\nabla_{\bfu_0} g_1(\bfu_0) && \in \Rbb^{n_0\times n_1}\\
        \nabla_{\bfu_0}^2 \bfu_1 &= \nabla_{\bfu_0}^2 g_1(\bfu_0) && \in \Rbb^{n_0\times n_0\times n_1}
    \end{align}
Then, we compute the derivatives of subsequent layers using the following mappings for $i=1, \dots, \ell-1$
    \begin{align}
        \nabla_{\bfu_0}\bfu_{i+1} &= \nabla_{\bfu_0} \bfu_i \nabla_{\bfu_i}g_{i+1}(\bfu_i) && \in \Rbb^{n_0\times n_{i+1}}\\
        \nabla_{\bfu_0}^2\bfu_{i+1} &= \nabla_{\bfu_i}^2g_{i+1}(\bfu_i) \times_1 \nabla_{\bfu_0}\bfu_{i} \times_2 \nabla_{\bfu_0}\bfu_{i}^\top \nonumber\\
	&\qquad + \nabla_{\bfu_0}^2\bfu_i \times_3 \nabla_{\bfu_i}g_{i+1}(\bfu_i) &&\in \Rbb^{n_0\times n_0 \times n_{i+1}}\label{eq:Hessian_ui}
    \end{align}
where $\times_k$ is the mode-$k$ product [@KoldaBader09] and $\nabla_{\bfu_0} \bfu_\ell \equiv \nabla_{\bfu_0} f(\bfu_0)$ is the Hessian we want to compute. 
The Hessian mapping in \autoref{eq:Hessian_ui} is illustrated in \autoref{fig:hessianIllustration}. For efficiency, we store $\nabla_{\bfu_{i}}  g_{i+1}(\bfu_{i})$ when we compute the gradient and re-use this matrix to form the Hessian.  
Notice that the sizes of the derivatives always depend on the number of input features, $n_{0}$. 

![Illustration of Hessian computation of $\nabla_{\bfu_0}^ 2\bfu_i$ in forward mode. Note that for the Gauss-Newton term, the gray three-dimensional array $\nabla_{\bfu_i} g_{i+1}(\bfu_i)$ is treated as a stack of matrices.  Then, the same Jacobian matrix $\nabla_{\bfu_0}\bfu_i$ is broadcast to each matrix in the stack, illustrated by the repeated cyan matrices. In the second term, the green matrix $\nabla_{\bfu_i}g_{i+1}(\bfu_i)$ is applied along the third dimension of the magenta three-dimensional array, $\nabla_{\bfu_0}\bfu_i$. Both of these operations can be efficiently parallelized, particularly on GPUs. \label{fig:hessianIllustration}](img/HessianIllustration.png){ width=80% }

## Computing Derivatives in Backward Mode

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
			 \qquad &&\in \Rbb^{n_{i-1} \times n_{i-1} \times n_{\ell}}, \label{eq:Hessian_ui2}
    \end{align}
where $\times_k$ is the mode-$k$ product \cite{koldabader}.  For efficiency, we re-use $nabla_{\bfu_{i-1}}  g_{i}(\bfu_{i-1})$ from the gradient computation to compute the Hessian.  Notice that the sizes of the derivatives always depend on the number of output features, $n_{\ell}$. 

## Forward Mode vs. Backward Mode

The computational efficiency of computing derivatives is proportional to the number of input features $n_0$ and the number of output features $n_{\ell}$.  The heuristic we use is if $n_0 < n_\ell$, we compute derivatives in forward mode, otherwise we compute derivatives in backward mode. Our implementation automatically selects the mode of derivative computation based on this heuristic. Users have the option to select their preferred mode of derivative computation if desired. 


## Testing Derivative Implementations
The `hessQuik` package includes methods to test derivative implementations and corresponding unit tests.  The main test we use is a Taylor approximation test; for details, see [@haberDerivative]. 


# Efficiency of `hessQuik`

We compare the time to compute the Hessian of a neural network with respect to the input features of three approaches: 'hessQuik' (AD-free method), 'PytorchAD' which uses automatic differentiation following the implementation in [@huang2021convex], and `PytorchHessian' which uses the built-in Pytorch [Hessian function](https://pytorch.org/docs/stable/generated/torch.autograd\
.functional.hessian.html).


## Network Architecture

We compare the time to compute the gradient and Hessian of a network with an input dimension $d = 2^k$ where $k=0,1,\dots,10$.  We implement a residual neural network [@He2016:deep] with the width is $w=16$, the depth is $N=4$, and various numbers of output features, $n_\ell$.  For simplicity, same network architecture is used for every timing test. 

## Hardware
For reproducibility, we compare the time to compute the Hessian using Google Colaboratory (Colab) Pro [@googleColab] and provide the notebook in the repository. When using a CPU runtime, Google Colab Pro uses an Intel(R) Xeon(R) CPU with 2.20GHz processor base speed. When using a GPU runtime, Google Colab Pro uses a Tesla P100 with 16 GB of memory.  

## Results

We compare the performance of three approaches to compute Hessians of a neural network for two different cases: scalar outputs ($n_\ell = 1$) and vector-valued outputs ($n_\ell \ge 1$).  In our experiments, we see faster Hessian computations using `hessQuik` and noticeable acceleration on the GPU, particularly for networks with larger input and output dimensions. 

![Average time over $10$ trials to evaluate and compute the Hessian with respect to the input features for one output feature ($n_{\ell}=1$). Solid lines represent timings on the CPU and dashed lines are timings on the GPU. The circle markers are the timings obtained using `hessQuik` and produce the fastest timings for a various numbers of input features $n_0$. \label{fig:scalar}](img/hessQuik_timing_scalar.png){ width=80% }

![Average time over $10$ trials to compute the Hessian with respect to the input features with variable number of input and output features. Each row corresponds to a number of input features, $n_0$, each column corresponds to a number of output features, $n_{\ell}$, and color represents the amount of time to compute.The `hessQuik` timings remain relatively constant as the number of output features changes whereas the `PytorchAD` timings significantly increase as the number of output features increases.](img/hessQuik_timing_vector.png){ width=80% }

# Conclusions
We have presented `hessQuik`, a simple, user-friendly repository for computing second-order derivatives of neural networks and other models constructed via composition of functions.  Our implementation scales well with the number of output features and on GPUs and is faster than automatic-differentiation-based second-order derivative computations.

# Acknowledgements


# References