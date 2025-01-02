Developer Guide
===============

Welcome to the `raking` developer guide! This section provides instructions and tips for contributing to the package, including setting up your development environment, running tests, and updating the documentation.

.. toctree::
   :hidden:

   setup
   testing
   contributing_code
   contributing_docs

Setting Up Your Development Environment
---------------------------------------

To get started with developing for `raking`, follow the instructions below to set up your local environment.

1. **Clone the Repository**:

   Start by cloning the `raking` repository from GitHub:

   .. code-block:: bash

       git clone https://github.com/ihmeuw-msca/raking.git
       cd raking

2. **Create a Virtual Environment**:

   Create a Python virtual environment to isolate your development dependencies:

   .. code-block:: bash

       python -m venv .venv
       source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`

3. **Install Development Dependencies**:

   Install the package along with the development dependencies:

   .. code-block:: bash

       pip install -e ".[dev]"

   This command installs the package in editable mode (`-e`) along with the optional `dev` dependencies.

4. **Verify Installation**:

   Ensure the dependencies are installed correctly:

   .. code-block:: bash

       python -m pytest
       sphinx-build -b html docs docs/_build

   If these commands run without errors, you’re ready to start developing!

.. admonition:: Note
    :class: hint

    For details on contributing to the codebase, see :ref:`Contributing Code`.
    For documentation contributions, see :ref:`Contributing to Documentation`.
