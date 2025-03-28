import numpy as np
import pandas as pd

from set_inequality_problems import set_infant_mortality
from raking_inequality import raking_chi2_inequality

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

# Read the data
females = pd.read_csv('/ihme/homes/ducela/stash/raking/data/pre_raked_female_mx_EGY_20240216_094156_322.csv')
females['t'] = females['age_end'] - females['age_start']
females['age_group'] = func(females['age_start'], females['age_end'])
females['value'] = females['mx'] * females['worldpop']

# Loop on subsets
years = females.year.unique()
draws = females.draw.unique()

females_raked = []
for year in years:
    for draw in draws:
        females_loc = females.loc[(females.year==year)&(females.draw==draw)]
        df_01m = females_loc[['value', 'loc_id', 'mx', 't', 'worldpop', 'year', 'draw', 'age_group', 'gbd_loc_value']]. \
            loc[females_loc.age_group=='0 to 1m']
        df_01y = females_loc[['value', 'loc_id', 'mx', 't', 'worldpop', 'year', 'draw', 'age_group', 'gbd_loc_value']]. \
            loc[females_loc.age_group=='0 to 1y']
        # Remove rows with zero population
        df = pd.merge(df_01m, df_01y, on=['year', 'loc_id'])
        valid_loc_id = df['loc_id'].loc[(df.worldpop_x>0.0)&(df.worldpop_y>0.0)].tolist()
        df_01m = df_01m.loc[df_01m.loc_id.isin(valid_loc_id)].sort_values(by=['loc_id'])
        df_01y = df_01y.loc[df_01y.loc_id.isin(valid_loc_id)].sort_values(by=['loc_id'])
        # Get input to raking functions
        y1 = (df_01m['value']).to_numpy()
        n1 = (df_01m['worldpop']).to_numpy()
        t1 = df_01m['t'].mean()
        s1 = df_01m['gbd_loc_value'].mean()
        q1 = np.ones(len(y1))
        y2 = (df_01y['value']).to_numpy()
        n2 = (df_01y['worldpop']).to_numpy()
        t2 = df_01y['t'].mean()
        s2 = df_01y['gbd_loc_value'].mean()
        q2 = np.ones(len(y2))
        # Solve raking problem
        (y, A, s, C, c, q, l, h) = set_infant_mortality(n1, n2, t1, t2, y1, y2, s1, s2, q1, q2)
        res = raking_chi2_inequality(y, A, s, C, c, q)
        # Add results to data frame
        beta = res.x
        I = len(y1)
        beta1 = beta[0:I]
        beta2 = beta[I:(2 * I)]
        df_01m['raked_value'] = beta1
        df_01y['raked_value'] = beta2
        df_raked = pd.concat([df_01m, df_01y])
        females_raked.append(df_raked)

females_raked = pd.concat(females_raked)
females_raked.to_csv('/ihme/homes/ducela/stash/raking/scripts_dev/females_raked_draws.csv', index=False)

