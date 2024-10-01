from raking.run_raking import run_raking


def test_run_raking_1D(tmp_path, example_1D):
    run_raking(
        1,
        str(tmp_path),
        example_1D.df_obs,
        [example_1D.df_margin],
        ["var1"],
    )


def test_run_raking_2D(tmp_path, example_2D):
    run_raking(
        2,
        str(tmp_path),
        example_2D.df_obs,
        [example_2D.df_margins_1, example_2D.df_margins_2],
        ["var1", "var2"],
    )


def test_run_raking_3D(tmp_path, example_3D):
    run_raking(
        3,
        str(tmp_path),
        example_3D.df_obs,
        [
            example_3D.df_margins_1,
            example_3D.df_margins_2,
            example_3D.df_margins_3,
        ],
        ["var1", "var2", "var3"],
    )
