## Basics
The examples folder contains code to supplement the Google Colaboratory notebooks.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1GCUSR9fGhQ9PoqfPxv8qRfqf88_ibyUA?usp=sharing) Practical Use: Hermite Interpolation


We include a related script for Hermite interpolation and the function used in the notebook in the following files:
```python
ex_peaks_hermite.py
peaks.py
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1842TWdILPNhiqLMw9JMZjdZ6T-B6hzul?usp=sharing) Tutorial: Constructing and Testing ```hessQuik``` Layers

We include a simple example of a timing test in ```ex_timing_test_hessQuik.py``` and we include the runs from our paper in the Google Colab notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elizabethnewman/hessQuik/blob/main/hessQuik/examples/hessQuikTimingTest.ipynb) Timing Tests

These scripts are designed to be run on Google Colaboratory.  

If you are interested in running the tests locally, please use the following from the command line:
```python
python run_timing_test.py
```

For convenience, this script allows for various command line arguments.  Some examples include
```python
python run_timing_test.py --num-input 5 --num-output 4 --network-wrapper hessQuik
```
The above command runs the timing test with our hessQuik implementation for 5 different input feature sizes from $2^0$ to $2^4$ and 4 different output features sizes from $2^0$ to $2^3$.

The flags used in the paper experiments are available in the notebook:

[HessQuikTimingTestLocal.ipynb](https://github.com/elizabethnewman/hessQuik/blob/main/hessQuik/examples/hessQuikTimingTestLocal.ipynb)

