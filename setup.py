from setuptools import setup, find_packages

setup(
    name='hessQuik',
    version='0.0.1',
    packages=['hessQuik'],
    url='https://github.com/elizabethnewman/hessQuik',
    license='MIT',
    author='Elizabeth Newman',
    author_email='elizabeth.newman@emory.edu',
    description='AD-free gradient and Hessian computations',
    install_requires=['torch>=1.10.0', 'numpy>=1.20.0'],
    extras_require={'interactive': ['matplotlib>=2.2.0']}
)
