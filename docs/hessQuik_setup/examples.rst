Examples
========

Hermite Interpolation
---------------------

Traditoinal polynomial interpolation seeks to find a polynomial to approximate an underlying function at given points and correspoonding function values.
`Hermite interpolation`_ seeks a polynomial that additionally fits derivative values at the given points.
Each given point requires more information, but fewer points are required to form a quality polynomial approximation.

``hessQuik`` makes it easy to obtain first- and second-order derivative information for the inputs of a network,
and hence is well-suited for fitting values and derivatives.

Check out this `Google Colab notebook for Hermite interpolation`_ to see ``hessQuik`` fit the :py:func:`hessQuik.utils.data.peaks` function using derivative information!

.. _Google Colab notebook for Hermite interpolation: https://colab.research.google.com/github/elizabethnewman/hessQuik/blob/main/hessQuik/examples/hessQuikPeaksHermiteInterpolation.ipynb
.. _Hermite interpolation: https://en.wikipedia.org/wiki/Hermite_interpolation

Testing New Layers
------------------

``hessQuik`` provides tools to develop and test new layers.
The package provides testing tools to ensure the derivatives are implemented correctly.
Choosing the best implementation of a given layer requires taking timing and storage costs into account.

Check out this `Google Colab notebook on testing layers`_ to see various implementations of the :py:class::`hessQuik.layers.single_layer.singleLayer` and testing methods!

.. _Google Colab notebook on testing layers: https://colab.research.google.com/github/elizabethnewman/hessQuik/blob/main/hessQuik/examples/hessQuikSingleLayerTutorial.ipynb



