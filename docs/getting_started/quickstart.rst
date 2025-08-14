==========
Quickstart
==========

Example 1
---------

This is how to use the raking package when we do not have draws and we do not want the covariance matrix of the raked values.

.. code-block:: python

    import pandas as pd
    from raking.run_raking import run_raking

    YOUR_PATH = '/Users/ducela/Documents/Raking/ihmeuw-msca/raking/'

    # 1D example
    df_obs = pd.read_csv(YOUR_PATH + 'tests/examples/example_1D/observations.csv')
    df_margin = pd.read_csv(YOUR_PATH + 'tests/examples/example_1D/margin.csv')
    (df_raked, dummy1, dummy2, dummy3) = run_raking(
        1,
        df_obs,
        [df_margin],
        ['var1'],
        cov_mat=False
    )

    # 2D example
    df_obs = pd.read_csv(YOUR_PATH + 'tests/examples/example_2D/observations.csv')
    df_margins_1 = pd.read_csv(YOUR_PATH + 'tests/examples/example_2D/margins_1.csv')
    df_margins_2 = pd.read_csv(YOUR_PATH + 'tests/examples/example_2D/margins_2.csv')
    (df_raked, dummy1, dummy2, dummy3) = run_raking(
        2,
        df_obs,
        [df_margins_1, df_margins_2],
        ['var1', 'var2'],
        cov_mat=False
    )

    # 3D example
    df_obs = pd.read_csv(YOUR_PATH + 'tests/examples/example_3D/observations.csv')
    df_margins_1 = pd.read_csv(YOUR_PATH + 'tests/examples/example_3D/margins_1.csv')
    df_margins_2 = pd.read_csv(YOUR_PATH + 'tests/examples/example_3D/margins_2.csv')
    df_margins_3 = pd.read_csv(YOUR_PATH + 'tests/examples/example_3D/margins_3.csv')
    (df_raked, dummy1, dummy2, dummy3) = run_raking(
        3,
        df_obs,
        [df_margins_1, df_margins_2, df_margins_3],
        ['var1', 'var2', 'var3'],
        cov_mat=False
    )

    # USHD example (level 1 to level 0)
    df_obs = pd.read_csv(YOUR_PATH + 'tests/examples/example_USHD/observations.csv')
    df_margin = pd.read_csv(YOUR_PATH + 'tests/examples/example_USHD/margins.csv')
    (df_raked, dummy1, dummy2, dummy3) = run_raking(
        'USHD',
        df_obs,
        [df_margin],
        None,
        ['_all', 1, 0],
        cov_mat=False,
    )

    # USHD example (level 2 to level 1)
    df_obs = pd.read_csv(YOUR_PATH + 'tests/examples/example_USHD_lower/observations.csv')
    df_margin_cause = pd.read_csv(YOUR_PATH + 'tests/examples/example_USHD_lower/margins_cause.csv')
    df_margin_county = pd.read_csv(YOUR_PATH + 'tests/examples/example_USHD_lower/margins_county.csv')
    df_margin_all_causes = pd.read_csv(YOUR_PATH + 'tests/examples/example_USHD_lower/margins_all_causes.csv')
    (df_raked, dummy1, dummy2, dummy3) = run_raking(
        'USHD_lower',
        df_obs,
        [df_margin_cause, df_margin_county, df_margin_all_causes],
        None,
        ['_inj', 1, 0],
        cov_mat=False,
    )

This is how to use the raking package when we have draws and we want the covariance matrix of the raked values.

.. code-block:: python

    import pandas as pd
    from raking.run_raking import run_raking

    YOUR_PATH = '/Users/ducela/Documents/Raking/ihmeuw-msca/raking/'

    # 1D example
    df_obs = pd.read_csv(YOUR_PATH + 'tests/examples/example_1D_draws/observations.csv')
    df_margin = pd.read_csv(YOUR_PATH + 'tests/examples/example_1D_draws/margin.csv')
    (df_raked, dummy1, dummy2, sigma) = run_raking(
        1,
        df_obs,
        [df_margin],
        ['var1'],
        draws='draws',
        cov_mat=True,
    )

    # 2D example
    df_obs = pd.read_csv(YOUR_PATH + 'tests/examples/example_2D_draws/observations.csv')
    df_margins_1 = pd.read_csv(YOUR_PATH + 'tests/examples/example_2D_draws/margins_1.csv')
    df_margins_2 = pd.read_csv(YOUR_PATH + 'tests/examples/example_2D_draws/margins_2.csv')
    (df_raked, dummy1, dummy2, sigma) = run_raking(
        2,
        df_obs,
        [df_margins_1, df_margins_2],
        ['var1', 'var2'],
        draws='draws',
        cov_mat=True,
    )

    # 3D example
    df_obs = pd.read_csv(YOUR_PATH + 'tests/examples/example_3D_draws/observations.csv')
    df_margins_1 = pd.read_csv(YOUR_PATH + 'tests/examples/example_3D_draws/margins_1.csv')
    df_margins_2 = pd.read_csv(YOUR_PATH + 'tests/examples/example_3D_draws/margins_2.csv')
    df_margins_3 = pd.read_csv(YOUR_PATH + 'tests/examples/example_3D_draws/margins_3.csv')
    (df_raked, dummy1, dummy2, dummy3) = run_raking(
        3,
        df_obs,
        [df_margins_1, df_margins_2, df_margins_3],
        ['var1', 'var2', 'var3'],
        draws='draws',
        cov_mat=True,
    )

    # USHD example (level 1 to level 0)
    df_obs = pd.read_csv(YOUR_PATH + 'tests/examples/example_USHD_draws/observations.csv')
    df_margin = pd.read_csv(YOUR_PATH + 'tests/examples/example_USHD_draws/margins.csv')
    (df_raked, dummy1, dummy2, sigma) = run_raking(
        'USHD',
        df_obs,
        [df_margin],
        None,
        ['_all', 1, 0],
        draws='draws',
        cov_mat=True,
    )
