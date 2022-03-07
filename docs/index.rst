.. Elizabeth Newman and Lars Ruthotto documentation master file, created by
   sphinx-quickstart on Sat Feb 26 12:45:02 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to hessQuik's documentation!
====================================

A lightweight package for fast, GPU-accelerated computation of gradients and Hessians of functions constructed via composition.

Statement of Need
-----------------

Deep neural networks (DNNs) and other composition-based models have become a staple of data science, garnering state-of-the-art results and gaining widespread use in the scientific community, particularly as surrogate models to replace expensive computations. The unrivaled universality and success of DNNs is due, in part, to the convenience of automatic differentiation (AD) which enables users to compute derivatives of complex functions without an explicit formula. Despite being a powerful tool to compute first-order derivatives (gradients), AD encounters computational obstacles when computing second-order derivatives (Hessians).

Knowledge of second-order derivatives is paramount in many growing fields and can provide insight into the optimization problem solved to build a good model. Hessians are notoriously challenging to compute efficiently with AD and cumbersome to derive and debug analytically. Hence, many algorithms approximate Hessian information, resulting in suboptimal performance. To address these challenges, hessQuik computes Hessians analytically and efficiently with an implementation that is accelerated on GPUs.

.. toctree::
   :maxdepth: 1
   :caption: Overview
   :glob:

   hessQuik_setup/getting_started
   hessQuik_setup/install
   hessQuik_setup/contributing
   hessQuik_setup/examples


.. toctree::
   :maxdepth: 2
   :caption: Documentation
   :glob:

   hessQuik_functionality/index
   hessQuik_derivations/index


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
