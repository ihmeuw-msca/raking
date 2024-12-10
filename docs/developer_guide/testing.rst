Running Tests
=============

`raking` uses `pytest` for testing. To ensure all tests pass, follow these steps:

1. Run the test suite:

   .. code-block:: bash

       pytest

2. Generate a test coverage report (optional):

   .. code-block:: bash

       pytest --cov=raking

3. View detailed coverage in HTML format:

   .. code-block:: bash

       pytest --cov=raking --cov-report=html

   Open the `htmlcov/index.html` file in your browser to inspect test coverage.

Remember to write tests for new features or bug fixes!
