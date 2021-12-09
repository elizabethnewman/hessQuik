from setuptools import setup, find_packages
# from glob import glob

setup(
    name='hessQuik',
    version='0.0.1',
    packages=find_packages(),
    scripts=['examples/hello_world.py'],
    url='https://github.com/elizabethnewman/hessQuik',
    license='MIT',
    author='Elizabeth Newman',
    author_email='elizabeth.newman@emory.edu',
    description='AD-free gradient and Hessian computations',
    install_requires=['torch', 'numpy'],
    extras_require={'interactive': ['matplotlib']}
)
