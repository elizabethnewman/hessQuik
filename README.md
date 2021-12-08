# hessQuik


## Installation

Create virtual environment

```html
virtualenv -p python env_name
source env_name/bin/activate
```


Install package

[comment]: <> (https://adamj.eu/tech/2019/03/11/pip-install-from-a-git-repository/)
```html
python -m pip install git+https://github.com/elizabethnewman/hessQuik.git
```
If the repository is private, use
```html
python -m pip install git+ssh://git@github.com/elizabethnewman/hessQuik.git
```

Make sure to import torch before importing hessQuik (this is a bug currently)

If hessQuik updated, reinstall via one of the following:
```html
pip install --upgrade --force-reinstall <package>
pip install -I <package>
pip install --ignore-installed <package>
```

When finished, deactivate virtual environment.

```html
deactivate
```

## Test Importing
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1GCUSR9fGhQ9PoqfPxv8qRfqf88_ibyUA?usp=sharing) Peaks Hermite Interpolation