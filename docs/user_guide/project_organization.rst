Project organization
====================

.. admonition:: Working in progress...
    :class: Attention

    Explain what different files and folders in the repo mean...

Here we describe what each file and folder in the project means and provide references and examples.

* :code:`.github/workflows`: GitHub Actions workflows for continuous integration and deployment.
    * For more please check :ref:`CI/CD`.
* :code:`.vscode`: VSCode configurations.
    * We use `Visual Studio Code <https://code.visualstudio.com>`_ as our main IDE.
    * `Getting Started with Python in VS Code <https://code.visualstudio.com/docs/python/python-tutorial>`_.
* :code:`.gitignore`: Files and folders that are ignored by Git.
    * For formal documentation please check `here <https://git-scm.com/docs/gitignore>`_.
    * For more please check `the list of common gitignore <https://github.com/github/gitignore>`_.
    * We use the `gitignore extension <https://marketplace.visualstudio.com/items?itemName=codezombiech.gitignore>`_ in VSCode to conveniently genterate gitignore file.
* :code:`src`: Source code of the project.
    * For a minimum example of the source folder, please check `A simple project <https://packaging.python.org/en/latest/tutorials/packaging-projects/#a-simple-project>`_.
* :code:`docs`: Documentation of the project.
    * We use `Sphinx <https://www.sphinx-doc.org/en/master/>`_ to build documentation.
    * For more please check :ref:`Documentation`.
* :code:`tests`: Tests of the project.
    * We use `Pytest <https://docs.pytest.org/en/stable/>`_ as the testing framework.
    * For more please check :ref:`Testing`.
* :code:`pyproject.toml`: Metadata of the project. This is enssential for Python packaging.
    * For more information please check `this guide <https://packaging.python.org/en/latest/guides/writing-pyproject-toml/>`_.
* :code:`ruff.toml`: Configuration for `Ruff <https://docs.astral.sh/ruff/>`_ linter and formatter.
    * For more please check :ref:`Style guide`.
    * We use `Ruff extension <https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff>`_ in VSCode to lint and format the code.
* :code:`LICENSE`: License of the project. Here are some references
    * `Licensing a repository <https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository>`_.
    * `Adding a license to a repository on GitHub <https://docs.github.com/en/communities/setting-up-your-project-for-healthy-contributions/adding-a-license-to-a-repository>`_.
    * Adding a license on VSCode with extention `Choose a License extension <https://marketplace.visualstudio.com/items?itemName=ultram4rine.vscode-choosealicense>`_.
* :code:`README.md`: Main page of the project including
    * Brief description of the project.
    * Installation instructions.
    * A quick example.
* :code:`CODE_OF_CONDUCT.md`: Code of conduct of the project.
    * `Adding a code of conduct to your project <https://docs.github.com/en/communities/setting-up-your-project-for-healthy-contributions/adding-a-code-of-conduct-to-your-project>`_.
