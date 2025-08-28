=======================
Quickstart experimental
=======================

Example 1
---------

This is how to use the experimental version of the raking package. It does not contain uncertainty propagation yet.

.. code-block:: python

    import pandas as pd
    from raking.experimental import DataBuilder
    from raking.experimental import DualSolver

    YOUR_PATH = '/Users/ducela/Documents/Raking/ihmeuw-msca/raking/'

    # 1D example
    df_obs = pd.read_csv(YOUR_PATH + "tests/examples/example_1D/observations.csv")
    df_margin = pd.read_csv(YOUR_PATH + "tests/examples/example_1D/margin.csv")
    df_obs['weights'] = 1.0
    df_margin['var1'] = -1
    df_margin['weights'] = np.inf
    df_margin.rename(columns={'value_agg_over_var1': 'value'}, inplace=True)
    df = pd.concat([df_obs, df_margin])
    data_builder = DataBuilder(
        dim_specs={'var1': -1},
        value='value',
        weights='weights'
    )
    data = data_builder.build(df)
    solver = DualSolver(distance='entropic', data=data)
    df_raked = solver.solve()

    # 2D example
    df_obs = pd.read_csv(YOUR_PATH + "tests/examples/example_2D/observations.csv")
    df_margins_1 = pd.read_csv(
        YOUR_PATH + "tests/examples/example_2D/margins_1.csv"
    )
    df_margins_2 = pd.read_csv(
        YOUR_PATH + "tests/examples/example_2D/margins_2.csv"
    )
    df_obs['weights'] = 1.0
    df_margins_1['var1'] = -1
    df_margins_1['weights'] = np.inf
    df_margins_1.rename(columns={'value_agg_over_var1': 'value'}, inplace=True)
    df_margins_2['var2'] = -1
    df_margins_2['weights'] = np.inf
    df_margins_2.rename(columns={'value_agg_over_var2': 'value'}, inplace=True)
    df = pd.concat([df_obs, df_margins_1, df_margins_2])
    data_builder = DataBuilder(
        dim_specs={'var1': -1, 'var2': -1},
        value='value',
        weights='weights'
    )
    data = data_builder.build(df)
    solver = DualSolver(distance='entropic', data=data)
    df_raked = solver.solve()

    # 3D example
    df_obs = pd.read_csv(YOUR_PATH + "tests/examples/example_3D/observations.csv")
    df_margins_1 = pd.read_csv(
        YOUR_PATH + "tests/examples/example_3D/margins_1.csv"
    )
    df_margins_2 = pd.read_csv(
        YOUR_PATH + "tests/examples/example_3D/margins_2.csv"
    )
    df_margins_3 = pd.read_csv(
        YOUR_PATH + "tests/examples/example_3D/margins_3.csv"
    )
    df_obs['weights'] = 1.0
    df_margins_1['var1'] = -1
    df_margins_1['weights'] = np.inf
    df_margins_1.rename(columns={'value_agg_over_var1': 'value'}, inplace=True)
    df_margins_2['var2'] = -1
    df_margins_2['weights'] = np.inf
    df_margins_2.rename(columns={'value_agg_over_var2': 'value'}, inplace=True)
    df_margins_3['var3'] = -1
    df_margins_3['weights'] = np.inf
    df_margins_3.rename(columns={'value_agg_over_var3': 'value'}, inplace=True)
    df = pd.concat([df_obs, df_margins_1, df_margins_2, df_margins_3])
    data_builder = DataBuilder(
        dim_specs={'var1': -1, 'var2': -1, 'var3': -1},
        value='value',
        weights='weights'
    )
    data = data_builder.build(df)
    solver = DualSolver(distance='entropic', data=data)
    df_raked = solver.solve()

    # USHD example (level 1 to level 0)
    df_obs = pd.read_csv(YOUR_PATH + "tests/examples/example_USHD/observations.csv")
    df_margin = pd.read_csv(YOUR_PATH + "tests/examples/example_USHD/margins.csv")
    df_obs['weights'] = 1.0
    df_obs.replace({'cause': '_all', 'race': 1}, -1, inplace=True)
    df_obs.drop(columns=['upper'], inplace=True)
    df_obs.replace({'cause': {'_comm': 1, '_inj': 2, '_ncd': 3}}, inplace=True)
    df_margin['race'] = -1
    df_margin['county'] = -1
    df_margin['weights'] = np.inf
    df_margin.rename(columns={'value_agg_over_race_county': 'value'}, inplace=True)
    df_margin.replace({'cause': {'_all': -1, '_comm': 1, '_inj': 2, '_ncd': 3}}, inplace=True)
    df = pd.concat([df_obs, df_margin])
    df = df.astype({'cause': 'int64'})
    data_builder = DataBuilder(
        dim_specs={'cause': -1, 'race': -1, 'county': -1},
        value='value',
        weights='weights'
    )
    data = data_builder.build(df)
    solver = DualSolver(distance='entropic', data=data)
    df_raked = solver.solve()

    # USHD example (level 2 to level 1)
    df_obs = pd.read_csv(
        YOUR_PATH + "tests/examples/example_USHD_lower/observations.csv"
    )
    df_margin_cause = pd.read_csv(
        YOUR_PATH + "tests/examples/example_USHD_lower/margins_cause.csv"
    )
    df_margin_county = pd.read_csv(
        YOUR_PATH + "tests/examples/example_USHD_lower/margins_county.csv"
    )
    df_margin_all_causes = pd.read_csv(
        YOUR_PATH + "tests/examples/example_USHD_lower/margins_all_causes.csv"
    )
    df_obs['weights'] = 1.0
    df_obs.replace({'race': 1}, -1, inplace=True)
    df_obs.drop(columns=['upper'], inplace=True)
    df_obs.replace({'cause': {'_intent': 1, '_unintent': 2, 'inj_trans': 3}}, inplace=True)
    df_margin_cause['race'] = -1
    df_margin_cause['county'] = -1
    df_margin_cause['weights'] = np.inf
    df_margin_cause.rename(columns={'value_agg_over_race_county': 'value'}, inplace=True)
    df_margin_cause.replace({'cause': {'_intent': 1, '_unintent': 2, 'inj_trans': 3}}, inplace=True)
    df_margin_county['cause'] = -1
    df_margin_county['race'] = -1
    df_margin_county['weights'] = np.inf
    df_margin_county.rename(columns={'value_agg_over_cause_race': 'value'}, inplace=True)
    df_margin_all_causes['cause'] = -1
    df_margin_all_causes['weights'] = np.inf
    df_margin_all_causes.rename(columns={'value_agg_over_cause': 'value'}, inplace=True)
    df = pd.concat([df_obs, df_margin_cause, df_margin_county, df_margin_all_causes])
    df = df.astype({'cause': 'int64'})
    data_builder = DataBuilder(
        dim_specs={'cause': -1, 'race': -1, 'county': -1},
        value='value',
        weights='weights'
    )
    data = data_builder.build(df)
    solver = DualSolver(distance='entropic', data=data)
    df_raked = solver.solve()
