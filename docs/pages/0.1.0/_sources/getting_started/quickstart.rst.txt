==========
Quickstart
==========

Example 1
---------

This is how to use the raking packkage when we do not have draws and we do not want the covariance matrix of the raked values.

.. code-block:: python

    from raking.run_raking import run_raking

    # 1D example
    df_obs = pd.read_csv('YOUR_PATH/tests/examples/example_1D/observations.csv')
    df_margin = pd.read_csv('YOUR_PATH/tests/examples/example_1D/margin.csv')

    (df_raked, dummy1, dummy2, dummy3) = run_raking(1, df_obs, [df_margin], ['var1'], cov_mat=False)

    # 2D example
    df_obs = pd.read_csv('YOUR_PATH/tests/examples/example_2D/observations.csv')
    df_margins_1 = pd.read_csv('YOUR_PATH/tests/examples/example_2D/margins_1.csv')
    df_margins_2 = pd.read_csv('YOUR_PATH/tests/examples/example_2D/margins_2.csv')

    (df_raked, dummy1, dummy2, dummy3) = run_raking(2, df_obs, [df_margins_1, df_margins_2], ['var1', 'var2'], cov_mat=False)

    # 3D example
    df_obs = pd.read_csv('YOUR_PATH/tests/examples/example_3D/observations.csv')
    df_margins_1 = pd.read_csv('YOUR_PATH/tests/examples/example_3D/margins_1.csv')
    df_margins_2 = pd.read_csv('YOUR_PATH/tests/examples/example_3D/margins_2.csv')
    df_margins_3 = pd.read_csv('YOUR_PATH/tests/examples/example_3D/margins_3.csv')

    (df_raked, dummy1, dummy2, dummy3) = run_raking(3, df_obs, [df_margins_1, df_margins_2, df_margins-3], ['var1', 'var2', 'var3'], cov_mat=False)