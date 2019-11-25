#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 09:40:13 2019

@author: inaki
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.close('all')

"""
 Read
"""

fname_mc = r'results_mc_2.txt'
fname_ga = r'results_ga.txt'
fname_sa = r'results_sa_2.txt'

colnames = ['i', 'dry_peat_vol', 'ndams', 'niter', 'days', 'day_week', 'month', 'day_month', 'time', 'yr' ]
rename_cols_sa = {'i':'dry_peat_vol', 'dry_peat_vol':'ndams', 'ndams':'niter', 'niter':'days', 'days':'day_week', 'day_week':'month', 'month':'day_month', 'day_month':'time', 'time':'yr' }
colnames_ga = {'dry_peat_vol', 'ndams', 'niter', 'days', 'day_week', 'month', 'day_month', 'time', 'yr', 'blocks', 'a', 'b', 'c'}
rename_cols_ga = {'day_month':'dry_peat_vol', 'niter':'ndams', 'c': 'niter', 'b':'days', 'blocks':'day_week', 'day_week':'month', 'days':'day_month', 'month':'time', 'ndams':'yr', 'a': 'b1', 'time':'b2', 'yr':'b3', 'dry_peat_vol':'b4'}

mc_df = pd.read_csv(fname_mc, delim_whitespace=True, header=None, names=colnames)
ga_df = pd.read_csv(fname_ga, delim_whitespace=True, header=None, names=colnames_ga)
sa_df = pd.read_csv(fname_sa, delim_whitespace=True, header=None, names=colnames)

"""
 Get info out
"""

dry_peat_vol_no_dams = 40191.730578848255 # normalization value


mc_df = mc_df[mc_df.i != 0]
number_dams = (2,4,6,8,10,12,14,16,18,20, 30, 40, 50)
mean_mc = [mc_df[mc_df.ndams == i]['dry_peat_vol'].mean()/dry_peat_vol_no_dams*100 for i in number_dams]
max_mc = [mc_df[mc_df.ndams == i]['dry_peat_vol'].max()/dry_peat_vol_no_dams*100 for i in number_dams]
min_mc = [mc_df[mc_df.ndams == i]['dry_peat_vol'].min()/dry_peat_vol_no_dams*100 for i in number_dams]

sa_df = sa_df.rename(columns=rename_cols_sa)
sa_df.dry_peat_vol = sa_df.dry_peat_vol/dry_peat_vol_no_dams*100

ga_df = ga_df.rename(columns=rename_cols_ga)
ga_df.dry_peat_vol = ga_df.dry_peat_vol/dry_peat_vol_no_dams*100

# Choose to plot ndams from number_dams above
sa_plot = sa_df.loc[sa_df['ndams'].isin(number_dams)]
ga_plot = ga_df.loc[ga_df['ndams'].isin(number_dams)]

"""
 Plot
"""

fig, ax = plt.subplots(1)
ax.plot(number_dams, mean_mc, alpha=1.0, color='red', label='random mean')
ax.fill_between(number_dams, max_mc, min_mc, facecolor='pink', alpha=0.7, label='random range')
ax.set_xlabel('Number of dams')
ax.set_ylabel('Volume fraction of dry peat (%)')

ax.scatter(x=sa_plot.ndams.to_numpy(), y=sa_plot.dry_peat_vol.to_numpy(), label='SA', alpha=0.8, color='orange')
ax.scatter(x=ga_plot.ndams.to_numpy(), y=ga_plot.dry_peat_vol.to_numpy(), label='GA', alpha=0.5, color='blue')
plt.legend()

fname_fig = r'results_plot.png'
plt.savefig(fname_fig)

plt.figure()
sns.set(style="whitegrid", palette="pastel", color_codes=True)

# Draw a nested violinplot and split the violins for easier comparison
sns.violinplot(x=mc_df.ndams, y=mc_df.dry_peat_vol/dry_peat_vol_no_dams,  inner="quart")

sns.despine(left=True)

plt.show()
