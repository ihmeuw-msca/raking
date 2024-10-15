from raking.run_raking import run_raking

def test_run_raking_1D(example_1D):
    (df_obs, Dphi_y, Dphi_s, sigma) = run_raking(
        dim=1,
        df_obs=example_1D.df_obs,
        df_margins=[example_1D.df_margin],
        var_names=['var1'],
        cov_mat=False
    )

def test_run_raking_2D(example_2D):
    (df_obs, Dphi_y, Dphi_s, sigma) = run_raking(
        dim=2,
        df_obs=example_2D.df_obs,
        df_margins=[
            example_2D.df_margins_1,
            example_2D.df_margins_2
        ],
        var_names=['var1', 'var2'],
        cov_mat=False
    )

def test_run_raking_3D(example_3D):
    (df_obs, Dphi_y, Dphi_s, sigma) = run_raking(
        dim=3,
        df_obs=example_3D.df_obs,
        df_margins=[
            example_3D.df_margins_1,
            example_3D.df_margins_2,
            example_3D.df_margins_3,
        ],
        var_names=['var1', 'var2', 'var3'],
        cov_mat=False
    )

