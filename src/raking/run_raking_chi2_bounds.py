# Import Python packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import functions for the raking with inequality constraints
from set_inequality_problems import set_infant_mortality
from raking_inequality import raking_chi2_inequality, raking_entropic_inequality, raking_logit_inequality

# Define age groups
def conditions(age_start, age_end):    
    if (age_start == 0) & (age_end == 0.0767123):
        age_group = '0 to 1m'
    elif (age_start == 0) & (age_end == 1.0):
        age_group = '0 to 1y'
    elif (age_start == 1) & (age_end == 5.0):
        age_group = '1 to 5y'
    else:
        age_group = 'error'
    return age_group
func = np.vectorize(conditions)

# Read the dataset
#males = pd.read_csv('/ihme/homes/ducela/stash/raking/data/pre_raked_male_mx_EGY_20240216_094156_322.csv')

# Compute probability of death
#males['t'] = males['age_end'] - males['age_start']
#males['qx'] = 1.0 - np.exp(- males['t'] * males['mx'])

# Assign age group
#males['age_group'] = func(males['age_start'], males['age_end'])

# Compute the mean and standard deviation of the draws
#males_mean = males.groupby(['year', 'loc_id', 'worldpop', 'age_group', 'gbd_loc_value']). \
#    agg({'mx': ['mean', 'std'], 't': 'mean', }).reset_index()
#males_mean.columns = [' '.join(col).strip() for col in males_mean.columns.values]
#males_mean['qx'] = 1.0 - np.exp(- males_mean['t mean'] * males_mean['mx mean'])

# Raking
#males_mean['value'] = males_mean['mx mean'] * males_mean['worldpop']
#males_mean['weight'] = males_mean['mx std'] * males_mean['worldpop']
#males_mean['lower'] = 0.0
#males_mean['upper'] = males_mean['worldpop']

#years = males_mean.year.unique()

# Raking with inequality constraints with weights (chi2 distance with bounds)
#males_raked_inequality_bounds = []
#for year in years:
    # Select values for year
#    males_loc = males_mean.loc[males_mean.year==year]
#    df_01m = males_loc[['value', 'loc_id', 'weight', 'mx mean', 't mean', 'worldpop', 'year', 'age_group', 'gbd_loc_value', 'upper', 'lower']]. \
#        loc[males_loc.age_group=='0 to 1m']
#    df_01y = males_loc[['value', 'loc_id', 'weight', 'mx mean', 't mean', 'worldpop', 'year', 'age_group', 'gbd_loc_value', 'upper', 'lower']]. \
#        loc[males_loc.age_group=='0 to 1y']
    # Remove rows with zero population
#    df = pd.merge(df_01m, df_01y, on=['year', 'loc_id'])
#    valid_loc_id = df['loc_id'].loc[(df.worldpop_x>0.0)&(df.worldpop_y>0.0)].tolist()
#    df_01m = df_01m.loc[df_01m.loc_id.isin(valid_loc_id)].sort_values(by=['loc_id'])
#    df_01y = df_01y.loc[df_01y.loc_id.isin(valid_loc_id)].sort_values(by=['loc_id'])
    # Get input to raking functions
#    y1 = df_01m['value'].to_numpy()
#    n1 = df_01m['worldpop'].to_numpy()
#    t1 = df_01m['t mean'].mean()
#    s1 = df_01m['gbd_loc_value'].mean()
#    q1 = df_01m['weight'].to_numpy()
#    l1 = df_01m['lower'].to_numpy()
#    h1 = df_01m['upper'].to_numpy()
#    y2 = df_01y['value'].to_numpy()
#    n2 = df_01y['worldpop'].to_numpy()
#    t2 = df_01y['t mean'].mean()
#    s2 = df_01y['gbd_loc_value'].mean()
#    q2 = df_01y['weight'].to_numpy()
#    l2 = df_01y['lower'].to_numpy()
#    h2 = df_01y['upper'].to_numpy()
    # Solve raking problem
#    (y, A, s, C, c, q, l, h) = set_infant_mortality(n1, n2, t1, t2, y1, y2, s1, s2, q1, q2, l1, l2, h1, h2)
#    res = raking_chi2_inequality(y, A, s, C, c, q, l, h)
    # Add results to data frame
#    beta = res.x
#    I = len(y1)
#    beta1 = beta[0:I]
#    beta2 = beta[I:(2 * I)]
#    df_01m['raked_value'] = beta1
#    df_01y['raked_value'] = beta2
#    df_raked = pd.concat([df_01m, df_01y])
#    males_raked_inequality_bounds.append(df_raked)
#males_raked_inequality_bounds = pd.concat(males_raked_inequality_bounds)

#males_raked_inequality_bounds['mx raked'] = males_raked_inequality_bounds['raked_value'] / \
#    males_raked_inequality_bounds['worldpop']
#males_raked_inequality_bounds['qx raked'] = 1.0 - np.exp(- males_raked_inequality_bounds['t mean'] * \
#    males_raked_inequality_bounds['mx raked'])

# Save output
#males_raked_inequality_bounds.to_csv('/ihme/homes/ducela/stash/raking/scripts_dev/males_raked_inequality_bounds.csv', index=False)

# Read the dataset
females = pd.read_csv('/ihme/homes/ducela/stash/raking/data/pre_raked_female_mx_EGY_20240216_094156_322.csv')

# Compute probability of death
females['t'] = females['age_end'] - females['age_start']
females['qx'] = 1.0 - np.exp(- females['t'] * females['mx'])

# Assign age group
females['age_group'] = func(females['age_start'], females['age_end'])

# Compute the mean and standard deviation of the draws
females_mean = females.groupby(['year', 'loc_id', 'worldpop', 'age_group', 'gbd_loc_value']). \
    agg({'mx': ['mean', 'std'], 't': 'mean', }).reset_index()
females_mean.columns = [' '.join(col).strip() for col in females_mean.columns.values]
females_mean['qx'] = 1.0 - np.exp(- females_mean['t mean'] * females_mean['mx mean'])

# Raking
females_mean['value'] = females_mean['mx mean'] * females_mean['worldpop']
females_mean['weight'] = females_mean['mx std'] * females_mean['worldpop']
females_mean['lower'] = 0.0
females_mean['upper'] = females_mean['worldpop']

years = females_mean.year.unique()

# Raking with inequality constraints with weights (chi2 distance with bounds)
females_raked_inequality_bounds = []
for year in years:
    # Select values for year
    females_loc = females_mean.loc[females_mean.year==year]
    df_01m = females_loc[['value', 'loc_id', 'weight', 'mx mean', 't mean', 'worldpop', 'year', 'age_group', 'gbd_loc_value', 'upper', 'lower']]. \
        loc[females_loc.age_group=='0 to 1m']
    df_01y = females_loc[['value', 'loc_id', 'weight', 'mx mean', 't mean', 'worldpop', 'year', 'age_group', 'gbd_loc_value', 'upper', 'lower']]. \
        loc[females_loc.age_group=='0 to 1y']
    # Remove rows with zero population
    df = pd.merge(df_01m, df_01y, on=['year', 'loc_id'])
    valid_loc_id = df['loc_id'].loc[(df.worldpop_x>0.0)&(df.worldpop_y>0.0)].tolist()
    df_01m = df_01m.loc[df_01m.loc_id.isin(valid_loc_id)].sort_values(by=['loc_id'])
    df_01y = df_01y.loc[df_01y.loc_id.isin(valid_loc_id)].sort_values(by=['loc_id'])
    # Get input to raking functions
    y1 = df_01m['value'].to_numpy()
    n1 = df_01m['worldpop'].to_numpy()
    t1 = df_01m['t mean'].mean()
    s1 = df_01m['gbd_loc_value'].mean()
    q1 = df_01m['weight'].to_numpy()
    l1 = df_01m['lower'].to_numpy()
    h1 = df_01m['upper'].to_numpy()
    y2 = df_01y['value'].to_numpy()
    n2 = df_01y['worldpop'].to_numpy()
    t2 = df_01y['t mean'].mean()
    s2 = df_01y['gbd_loc_value'].mean()
    q2 = df_01y['weight'].to_numpy()
    l2 = df_01y['lower'].to_numpy()
    h2 = df_01y['upper'].to_numpy()
    # Solve raking problem
    (y, A, s, C, c, q, l, h) = set_infant_mortality(n1, n2, t1, t2, y1, y2, s1, s2, q1, q2, l1, l2, h1, h2)
    res = raking_chi2_inequality(y, A, s, C, c, q, l, h)
    # Add results to data frame
    beta = res.x
    I = len(y1)
    beta1 = beta[0:I]
    beta2 = beta[I:(2 * I)]
    df_01m['raked_value'] = beta1
    df_01y['raked_value'] = beta2
    df_raked = pd.concat([df_01m, df_01y])
    females_raked_inequality_bounds.append(df_raked)
females_raked_inequality_bounds = pd.concat(females_raked_inequality_bounds)

females_raked_inequality_bounds['mx raked'] = females_raked_inequality_bounds['raked_value'] / \
    females_raked_inequality_bounds['worldpop']
females_raked_inequality_bounds['qx raked'] = 1.0 - np.exp(- females_raked_inequality_bounds['t mean'] * \
    females_raked_inequality_bounds['mx raked'])

# Save output
females_raked_inequality_bounds.to_csv('/ihme/homes/ducela/stash/raking/scripts_dev/females_raked_inequality_bounds.csv', index=False)

