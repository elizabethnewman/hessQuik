from setuptools import setup, find_packages
# from glob import glob

docs_extras = [
    'Sphinx >= 3.0.0',  # Force RTD to use >= 3.0.0
    'docutils',
    'pylons-sphinx-themes >= 1.0.8',  # Ethical Ads
    'pylons_sphinx_latesturl',
    'repoze.sphinx.autointerface',
    'sphinxcontrib-autoprogram',
]

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
    extras_require={'interactive': ['numpy', 'matplotlib'], 'docs': docs_extras}
)
