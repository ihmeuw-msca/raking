Installing raking
=================

Python Version
--------------

The `raking` package is written in Python and requires **Python 3.11** or later.

Installation Options
--------------------

You can install the `raking` package either **from PyPI** or **directly from the GitHub repository**. Choose the method that best fits your needs.

1. Installing from PyPI
   ----------------------

   Installing from PyPI is the recommended method for most users as it provides a straightforward installation process.

   a. **Basic Installation**

      To install the latest stable release of `raking` from PyPI, run:

      .. code-block:: bash

          pip install raking

   b. **Installing with Optional Dependencies**

      The `raking` package includes optional dependencies for **testing** and **documentation**. To install these extras, use the following command:

      .. code-block:: bash

          pip install raking[test,docs]

      **Explanation:**

      - **`raking`**: The name of the package.
      - **`[test,docs]`**: Specifies that you want to include both the `test` and `docs` optional dependencies as defined in the `pyproject.toml`.

      **Optional Dependencies:**

      - **`test`**: Includes packages like `pytest` and `pytest-cov` for testing.
      - **`docs`**: Includes packages like `sphinx`, `sphinx-autodoc-typehints`, and `furo` for generating documentation.

   c. **Installing in a Virtual Environment (Recommended)**

      It's a best practice to use a **virtual environment** to manage your project's dependencies, ensuring isolation from other projects and system-wide packages.

      **Steps:**

      1. **Create a Virtual Environment**

         .. code-block:: bash

             python3 -m venv venv

      2. **Activate the Virtual Environment**

         - **On Windows:**

           .. code-block:: bash

               venv\Scripts\activate

         - **On macOS and Linux:**

           .. code-block:: bash

               source venv/bin/activate

      3. **Upgrade `pip`**

         .. code-block:: bash

             pip install --upgrade pip

      4. **Install `raking` with Optional Dependencies**

         .. code-block:: bash

             pip install raking[test,docs]

2. Installing from GitHub (Development Version)
   ---------------------------------------------

   If you want to **contribute** to the development of `raking` or need the **latest changes** that are not yet released on PyPI, you can install the package directly from the GitHub repository.

   a. **Clone the Repository**

      First, clone the `raking` repository to your local machine:

      .. code-block:: bash

          git clone "https://github.com/ihmeuw-msca/raking.git"
          cd raking

   b. **Set Up a Virtual Environment**

      Creating a virtual environment ensures that dependencies are managed separately from your system Python.

      .. code-block:: bash

          python3 -m venv .venv
          source .venv/bin/activate  # On Windows, use `.\.venv\Scripts\activate`

   c. **Upgrade `pip` and Install Dependencies**

      .. code-block:: bash

          python3 -m pip install --upgrade pip
          python3 -m pip install ipykernel jupyterlab notebook

   d. **Install the Package in Editable Mode**

      Installing in editable mode allows you to make changes to the codebase and have them reflected immediately without reinstalling the package.

      .. code-block:: bash

          pip install -e .[test,docs]

   e. **Set Up the Jupyter Kernel**

      To use the `raking` package within Jupyter notebooks, install a dedicated IPython kernel:

      .. code-block:: bash

          python3 -m ipykernel install --user --name env_raking --display-name "env_raking"

      **Explanation:**

      - **`--name env_raking`**: The internal name for the kernel.
      - **`--display-name "env_raking"`**: The name that will appear in the Jupyter interface.

   f. **Launch Jupyter Notebook**

      Start Jupyter Notebook to begin using the `raking` package:

      .. code-block:: bash

          jupyter notebook

      **Instructions:**

      1. **Open the Notebook:**
         - Navigate to the `docs/getting_started.ipynb` notebook within the Jupyter interface.

      2. **Change the Kernel:**
         - In the top right corner of the notebook interface, click on **"Kernel"** > **"Change kernel"** > **"env_raking"**.

      3. **Run the Examples:**
         - You can now execute the code cells in the notebook using the `raking` package.

---
