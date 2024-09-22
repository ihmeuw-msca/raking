import pytest
import numpy as np
import os
import pandas as pd

from raking.run_raking import run_raking

def test_run_raking_1D():
    print(os.getcwd())
    df_obs = pd.read_csv('../examples/example_1D/observations.csv')
    df_margin = pd.read_csv('../examples/example_1D/margin.csv')
    run_raking(1, '../examples/example_1D', df_obs, [df_margin], ['var1'])

def test_run_raking_2D():
    print(os.getcwd())
    df_obs = pd.read_csv('../examples/example_2D/observations.csv')
    df_margins_1 = pd.read_csv('../examples/example_2D/margins_1.csv')
    df_margins_2 = pd.read_csv('../examples/example_2D/margins_2.csv')
    run_raking(2, '../examples/example_2D', df_obs, [df_margins_1, df_margins_2], ['var1', 'var2'])

def test_run_raking_3D():
    print(os.getcwd())
    df_obs = pd.read_csv('../examples/example_3D/observations.csv')
    df_margins_1 = pd.read_csv('../examples/example_3D/margins_1.csv')
    df_margins_2 = pd.read_csv('../examples/example_3D/margins_2.csv')
    df_margins_3 = pd.read_csv('../examples/example_3D/margins_3.csv')
    run_raking(3, '../examples/example_3D', df_obs, [df_margins_1, df_margins_2, df_margins_3], ['var1', 'var2', 'var3'])

