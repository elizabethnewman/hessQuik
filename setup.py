from setuptools import setup, find_packages
# from glob import glob

setup(
    name='hessQuik',
    version='0.0.1',
    packages=find_packages(),
    url='https://github.com/elizabethnewman/hessQuik',
    license='MIT',
    author='Elizabeth Newman',
    author_email='elizabeth.newman@emory.edu',
    description='AD-free gradient and Hessian computations',
    install_requires=['torch'],
    extras_require={'interactive': ['numpy', 'matplotlib']}
)
