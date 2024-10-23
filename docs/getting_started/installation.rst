=================
Installing raking
=================

Python version
--------------

The package :code:`raking` is written in Python and requires Python 3.11 or later.

This is a development version of the package and it is not yet available on PyPI. To install it locally, please use:

.. code::

    git clone "https://github.com/ihmeuw-msca/raking.git"
    cd raking
    python3 -m venv .venv
    source .venv/bin/activate
    python3 -m pip install --upgrade pip
    python3 -m pip install ipykernel
    python3 -m pip install jupyterlab
    python3 -m pip install notebook
    pip install -e .
    python3 -m ipykernel install --user --name env_raking --display-name "env_raking"

Then launch Jupyter notebook using:

.. code::

    jupyter notebook

Open the notebook docs/getting_started.ipynb and change the kernel to "env_raking" on the top right of the page. You can then run the examples.
