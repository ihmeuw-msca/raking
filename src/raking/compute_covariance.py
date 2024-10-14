
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None

def compute_covariance_obs(df_obs, var_names, draws):

    nsamples = len(df_obs[draws].unique())
    var_names.reverse()
    df = df_obs[['value'] + var_names + [draws]]
    df.sort_values(by=var_names + [draws], inplace=True)
    value = df['value'].to_numpy()
    X = np.reshape(value, shape=(nsamples, -1), order='F')
    Xmean = np.mean(X, axis=0)
    Xc = X - Xmean
    sigma_yy = np.matmul(np.transpose(Xc), Xc) / nsamples
    return sigma_yy

def compute_covariance_margins_1D(df_margins, var_names, draws):
    
    nsamples = len(df_margins[draws].unique())
    df = df_margins[['value_agg_over_' + var_names[0]] + [draws]]
    df.sort_values(by=[draws], inplace=True)
    value = df['value_agg_over_' + var_names[0]].to_numpy()
    X = np.reshape(value, shape=(nsamples, -1), order='F')
    Xmean = np.mean(X, axis=0)
    Xc = X - Xmean
    sigma_ss = np.matmul(np.transpose(Xc), Xc) / nsamples
    return sigma_ss

def compute_covariance_margins_2D(df_margins_1, df_margins_2, var_names, draws):
    
    nsamples = len(df_margins_1[draws].unique())
    df1 = df_margins_1[[var_names[1], 'value_agg_over_' + var_names[0], draws]]
    df1.sort_values(by=[var_names[1], draws], inplace=True)
    df2 = df_margins_2[[var_names[0], 'value_agg_over_' + var_names[1], draws]]
    df2.sort_values(by=[var_names[0], draws], inplace=True)
    value1 = df1['value_agg_over_' + var_names[0]].to_numpy()
    value2 = df2['value_agg_over_' + var_names[1]].to_numpy()
    value = np.concatenate((value1, value2))
    X = np.reshape(value, shape=(nsamples, -1), order='F')
    X = X[:, 0:-1]
    Xmean = np.mean(X, axis=0)
    Xc = X - Xmean
    sigma_ss = np.matmul(np.transpose(Xc), Xc) / nsamples
    return sigma_ss

def compute_covariance_margins_3D(df_margins_1, df_margins_2, df_margins_3, var_names, draws):
    
    nsamples = len(df_margins_1[draws].unique())
    var1 = df_margins_2[var_names[0]].unique().tolist()
    var2 = df_margins_1[var_names[1]].unique().tolist()
    var3 = df_margins_1[var_names[2]].unique().tolist()
    var1.sort()
    var2.sort()
    var3.sort()
    df1 = df_margins_1[[var_names[1], var_names[2], 'value_agg_over_' + var_names[0], draws]]
    df1 = df1.loc[(df1[var_names[1]].isin(var2[0:-1]) | ((df1[var_names[1]]==var2[-1]) & (df1[var_names[2]]==var3[-1]))]
    df1.sort_values(by=[var_names[2], var_names[1], draws], inplace=True)
    df2 = df_margins_2[[var_names[0], var_names[2], 'value_agg_over_' + var_names[1], draws]]
    df2 = df2.loc[df2[var_names[2]].isin(var3[0:-1])]
    df2.sort_values(by=[var_names[0], var_names[2], draws, inplace=True)
    df3 = df_margins_3[[var_names[0], var_names[1], 'value_agg_over_' + var_names[2], draws]]
    df3 = df3.loc[df3[var_names[0]].isin(var1[0:-1])]
    df3.sort_values(by=[var_names[1], var_names[0], draws, inplace=True)
    value1 = df1['value_agg_over_' + var_names[0]].to_numpy()
    value2 = df2['value_agg_over_' + var_names[1]].to_numpy()
    value3 = df3['value_agg_over_' + var_names[2]].to_numpy()
    value = np.concatenate((value1, value2, value3))
    X = np.reshape(value, shape=(nsamples, -1), order='F')
    Xmean = np.mean(X, axis=0)
    Xc = X - Xmean
    sigma_ss = np.matmul(np.transpose(Xc), Xc) / nsamples
    return sigma_ss

def compute_covariance_obs_margins_1D(df_obs, df_margins, var_names, draws):

    nsamples = len(df_obs[draws].unique())
    var_names.reverse()
    df_obs = df_obs[['value'] + var_names + [draws]]
    df_obs.sort_values(by=var_names + [draws], inplace=True)
    df_margins = df_margins[['value_agg_over_' + var_names[0]] + [draws]]
    df_margins.sort_values(by=[draws], inplace=True)
    value_obs = df_obs['value'].to_numpy()
    X = np.reshape(value_obs, shape=(nsamples, -1), order='F')
    value_margins = df_margins['value_agg_over_' + var_names[0]].to_numpy()
    Y = np.reshape(value_margins, shape=(nsamples, -1), order='F')
    Xmean = np.mean(X, axis=0)
    Ymean = np.mean(Y, axis=0)
    Xc = X - Xmean
    Yc = Y - Ymean
    sigma_ys = np.matmul(np.transpose(Xc), Yc) / nsamples
    return sigma_ys

def compute_covariance_obs_margins_2D(df_obs, df_margins_1, df_margins_2, var_names, draws):

    nsamples = len(df_obs[draws].unique())
    var_names.reverse()
    df_obs = df_obs[var_names + [draws]]
    df_obs.sort_values(by=var_names + [draws], inplace=True)
    df_margins_1 = df_margins_1[[var_names[1], 'value_agg_over_' + var_names[0], draws]]
    df_margins_1.sort_values(by=[var_names[1], draws], inplace=True)
    df_margins_2 = df_margins_2[[var_names[0], 'value_agg_over_' + var_names[1], draws]]
    df_margins_2.sort_values(by=[var_names[0], draws], inplace=True)
    value_obs = df_obs['value'].to_numpy()
    X = np.reshape(value_obs, shape=(nsamples, -1), order='F')
    value_margins_1 = df_margins_1['value_agg_over_' + var_names[0]].to_numpy()
    value_margins_2 = df_margins_2['value_agg_over_' + var_names[1]].to_numpy()
    value_margins = np.concatenate((value_margins_1, value_margins_2))
    Y = np.reshape(value_margins, shape=(nsamples, -1), order='F')
    Xmean = np.mean(X, axis=0)
    Ymean = np.mean(Y, axis=0)
    Xc = X - Xmean
    Yc = Y - Ymean
    sigma_ys = np.matmul(np.transpose(Xc), Yc) / nsamples
    return sigma_ys

def compute_covariance_obs_margins_3D(df_obs, df_margins_1, df_margins_2, df_margins_3, var_names, draws):

    nsamples = len(df_obs[draws].unique())
    var_names.reverse()
    df_obs = df_obs[var_names + [draws]]
    df_obs.sort_values(by=var_names + [draws], inplace=True)
    var1 = df_margins_2[var_names[0]].unique().tolist()
    var2 = df_margins_1[var_names[1]].unique().tolist()
    var3 = df_margins_1[var_names[2]].unique().tolist()
    var1.sort()
    var2.sort()
    var3.sort()
    df_margins_1 = df_margins_1[[var_names[1], var_names[2], 'value_agg_over_' + var_names[0], draws]]
    df_margins_1 = df_margins_1.loc[(df_margins_1[var_names[1]].isin(var2[0:-1]) | ((df_margins_1[var_names[1]]==var2[-1]) & (df_margins_1[var_names[2]]==var3[-1]))]
    df_margins_1.sort_values(by=[var_names[2], var_names[1], draws], inplace=True)
    df_margins_2 = df_margins_2[[var_names[0], var_names[2], 'value_agg_over_' + var_names[1], draws]]
    df_margins_2 = df_margins_2.loc[df_margins_2[var_names[2]].isin(var3[0:-1])]
    df_margins_2.sort_values(by=[var_names[0], var_names[2], draws, inplace=True)
    df_margins_3 = df_margins_3[[var_names[0], var_names[1], 'value_agg_over_' + var_names[2], draws]]
    df_margins_3 = df_margins_3.loc[df_margins_3[var_names[0]].isin(var1[0:-1])]
    df_margins_3.sort_values(by=[var_names[1], var_names[0], draws, inplace=True)
    value_obs = df_obs['value'].to_numpy()
    value_margins_1 = df_margins_1['value_agg_over_' + var_names[0]].to_numpy()
    value_margins_2 = df_margins_2['value_agg_over_' + var_names[1]].to_numpy()
    value_margins_3 = df_margins_3['value_agg_over_' + var_names[2]].to_numpy()
    value = np.concatenate((value_obs, value_margins_1, df_margins_2, df_margins_3))
    X = np.reshape(value, shape=(nsamples, -1), order='F')
    Xmean = np.mean(X, axis=0)
    Xc = X - Xmean
    sigma_ys = np.matmul(np.transpose(Xc), Xc) / nsamples
    return sigma_ys

def check_covariance(sigma_yy, sigma_ss, sigma_ys):
    """
    """
    sigma = np.concatenate(( \
        np.concatenate((sigma_yy, sigma_ys), axis=1), \
        np.concatenate((np.transpose(sigma_ys), sigma_ss), axis=1)), axis=0)
    valid = True
    if np.allclose(np.transpose(sigma), sigma, rtol, atol):
        valid = False
    if np.any(np.linalg.eig(sigma)[0] < 0.0):
        valid = False
    if not valid:
        sigma_yy = np.diag(np.diag(sigma_yy))
        sigma_ss = np.diag(np.diag(sigma_ss))
        sigma_ys = np.zeros(sigma_ys.shape)
    return sigma_yy, sigma_ss, sigma_ys

