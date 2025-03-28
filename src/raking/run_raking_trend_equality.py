import numpy as np
import os
import pandas as pd
import pyreadr

from get_draws.api import get_draws

from raking.run_raking import run_raking

# Parameters
edu = 1
state = 15
sexes = [1, 2, 3]
ages = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
years = np.arange(2000, 2020)

# Correspondance with GBD ids
location_ids = {1:523, 2:524, 4:525, 5:526, 6:527, 8:528, 9:529, 10:530, 11:531, 12:532, \
    13:533, 15:534, 16:535, 17:536, 18:537, 19:538, 20:539, 21:540, 22:541, 23:542, \
    24:543, 25:544, 26:545, 27:546, 28:547, 29:548, 30:549, 31:550, 32:551, 33:552, \
    34:553, 35:554, 36:555, 37:556, 38:557, 39:558, 40:559, 41:560, 42:561, 44:562, \
    45:563, 46:564, 47:565, 48:566, 49:567, 50:568, 51:569, 53:570, 54:571, 55:572, 56:573}

age_ids = {20:9, 25:10, 30:11, 35:12, 40:13, 45:14, 50:15, 55:16, 60:17, 65:18, 70:19, 75:20, 80:30}

# Read the population data
population_path = '/mnt/team/msca/pub/raking_use_cases/2024_10_09_13_52_29model154/imputation0/population.rds'
population = pyreadr.read_r(population_path)[None]
races = population.race.unique().tolist()

df_raked = []
# Loop on sex and age
for sex in sexes:
    for age in ages:

        # Get the population data
        if sex == 3:
            population_loc = population.groupby(['year', 'edu', 'race', 'mcnty', 'state', 'age', 'race_label', 'edu_label']). \
                agg({'pop': 'sum'}).reset_index()
            population_loc['sex'] = 3
            population_loc = population_loc.loc[ \
                (population_loc.edu == edu) & \
                (population_loc.state == state) & \
                (population_loc.age == age)]
        else:
            population_loc = population.loc[ \
                (population.edu == edu) & \
                (population.state == state) & \
                (population.sex == sex) & \
                (population.age == age)]
        population_counties = population_loc[['year', 'mcnty', 'race', 'pop']]
        counties = population_counties.mcnty.unique().tolist()
        population_state = population_loc.groupby(['year']).agg({'pop': 'sum'}).reset_index()

        # Get the obesity data
        df_county = []
        for year in years:
            for race in races:
                path = '/mnt/team/msca/pub/raking_use_cases/2024_10_09_13_52_29model154/imputation0/est/' + \
                    'est_mcnty_' + str(year) + '_' + str(sex) + '_' + str(int(race)) + '_' + str(edu) + '.rds'
                df_year_race = pyreadr.read_r(path)[None]
                df_year_race = df_year_race.loc[df_year_race.area.isin(counties)]
                df_year_race = df_year_race.loc[
                    (df_year_race.age == age)]
                df_year_race = df_year_race[['year', 'race', 'area', 'source_v2', 'pred_mean']]. \
                    groupby(['year', 'race', 'area']).agg({'pred_mean': 'mean'}).reset_index()
                df_year_race = df_year_race.merge(population_counties, how='inner', left_on=['year', 'race', 'area'], right_on=['year', 'race', 'mcnty'])
                df_county.append(df_year_race)
        df_county = pd.concat(df_county)
        df_county['value'] = df_county['pred_mean'] * df_county['pop']

        # Get the GBD data
        df_GBD = []
        for year in years:
            df_tmp = get_draws(gbd_id_type='modelable_entity_id', \
                               gbd_id=24743, \
                               location_id=location_ids[state], \
                               year_id=year, \
                               source='epi', \
                               age_group_id=[age_ids[age]], \
                               sex_id=sex,
                               release_id=16)
            df_tmp = df_tmp.set_index( \
                ['age_group_id', 'location_id', 'measure_id', 'modelable_entity_id', 'sex_id', 'year_id', 'metric_id', 'model_version_id']). \
                stack().reset_index().rename(columns={'level_8': 'draw', 0:'obesity'})
            df_GBD.append(df_tmp)
        df_GBD = pd.concat(df_GBD)
        if sex == 3:
            df_GBD_mean = df_GBD.groupby( \
                ['age_group_id', 'location_id', 'measure_id', 'modelable_entity_id', 'sex_id', 'year_id', 'metric_id']). \
                agg({'obesity': 'mean'}).reset_index()
        else:    
            df_GBD_mean = df_GBD.groupby( \
                ['age_group_id', 'location_id', 'measure_id', 'modelable_entity_id', 'sex_id', 'year_id', 'metric_id', 'model_version_id']). \
                agg({'obesity': 'mean'}).reset_index()
        df_GBD_mean = df_GBD_mean.merge(population_state, left_on=['year_id'], right_on=['year'])
        df_GBD_mean['value_agg_over_var1'] = df_GBD_mean['obesity'] * df_GBD_mean['pop']

        # Rake for each year 
        for year in years:
            df_obs = df_county.loc[df_county.year==year].reset_index(drop=True).reset_index().rename(columns={'index': 'var1'})
            df_margin = df_GBD_mean.loc[df_GBD_mean.year==year][['value_agg_over_var1']]
            result = run_raking(1, df_obs, [df_margin], ['var1'], cov_mat=False)[0]
            result['age'] = age
            result['sex'] = sex
            df_raked.append(result)

# Save in output file
df_raked = pd.concat(df_raked)
df_raked.to_csv('/ihme/homes/ducela/stash/raking/scripts_dev/obesity/raked_classic.csv', index=False)

