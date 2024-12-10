Setting Up Your Environment
===========================

Follow the steps below to set up a local development environment for `raking`.

1. Clone the repository:

   .. code-block:: bash

       git clone https://github.com/ihmeuw-msca/raking.git
       cd raking

2. Create a Python virtual environment:

   .. code-block:: bash

       python -m venv .venv
       source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`

3. Install the package and development dependencies:

   .. code-block:: bash

       pip install -e ".[dev]"

Congratulations! You’re ready to start contributing.
