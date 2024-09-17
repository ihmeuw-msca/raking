"""Module with methods to format the data for the raking methods"""

import numpy as np
import pandas as pd

def format_dataframe(df, var_names, var_value, agg_var_names, weights=None, l=None, h=None, rtol: float = 1e-05, 
    atol:float = 1e-08):
    """
    """
    assert isinstance(df, pd.DataFrame), \
        'The input data should be a pandas data frame.'
    assert len(df) >= 2, \
        'There should be at least to data points in the data frame.'
    assert isinstance(var_names, list), \
        'Please enter the names of the columns containing the values of the categorical variables as a list.'
    assert (len(var_names) >= 1) and (len(var_names) <= 3), \
        'You should have 1, 2 or 3 categorical variables.'
    for var_name in var_names:
        assert isinstance(var_name, str), \
            'The name of the categorical variable ', + str(var_name) + ' should be a string.'
        assert var_name in df.columns.tolist(), 
            'The column for the categorical variable ' + var_name + ' is missing from the data frame.'
    assert isinstance(var_value, str), \
        'The name of the column containing the values to be raked should be a string.'
    assert var_value in df.columns.tolist(), 
        'The column containing the values to be raked is missing from the data frame.'
    assert isinstance(agg_var_names, list), \
        'Please enter the names of the columns containing the values for the aggregated categories as a list.'
    assert len(var_names) == len(agg_var_names), \
        'The number of columns for the agregated values should be equal to the number of columns for the categorical variables.'
    for agg_var_name in agg_var_names:
        assert isinstance(agg_var_name, str), \
            'The name of the column for the aggregated values ', + str(agg_var_name) + ' should be a string.'
        assert agg_var_name in df.columns.tolist(), 
            'The column for the aggregated values ' + var_name + ' is missing from the data frame.'
    if weights is not None:
        assert isinstance(weigths, str), \
            'The name of the column containing the weights should be a string.'
        assert weights in df.columns.tolist(), 
            'The column containing the weights is missing from the data frame.'
    
    dim = len(var_name)
    I = len(df.var_names[0].unique())
    if dim >= 2:
        J = len(df.var_names[1].unique())
    if dim >= 3:
        K = len(df.var_names[2].unique())
    df.sort_values(by=var_names[::-1], inplace=True)
    y = df.var_value.to_numpy()
    q = df.weights.to_numpy()

