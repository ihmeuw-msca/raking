import numpy as np
import pandas as pd

from raking.run_raking import run_raking

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
age_groups = females.age_group.unique()
draws = females.draw.unique()

females_raked = []
for year in years:
    for age_group in age_groups:
        for draw in draws:
            females_loc = females.loc[(females.year==year)&(females.draw==draw)&(females.age_group==age_group)]
            df_obs = females_loc[['value', 'loc_id', 'mx', 't', 'worldpop', 'year', 'age_group', 'draw', 'gbd_loc_value']]
            df_margin = females_loc.groupby(['year', 'age_group', 'draw']).agg({'gbd_loc_value': 'mean'}).reset_index(). \
                rename(columns={'gbd_loc_value': 'value_agg_over_loc_id'})[['value_agg_over_loc_id']]
            (df_raked, dummy1, dummy2, dummy3) = run_raking(1, df_obs, [df_margin], ['loc_id'], cov_mat=False)
            females_raked.append(df_raked)

females_raked = pd.concat(females_raked)
females_raked.to_csv('/ihme/homes/ducela/stash/raking/scripts_dev/females_raked_classic_draws.csv', index=False)

