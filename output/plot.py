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
rename_cols = {'i':'dry_peat_vol', 'dry_peat_vol':'ndams', 'ndams':'niter', 'niter':'days', 'days':'day_week', 'day_week':'month', 'month':'day_month', 'day_month':'time', 'time':'yr' }

mc_df = pd.read_csv(fname_mc, delim_whitespace=True, header=None, names=colnames)
ga_df = pd.read_csv(fname_ga, delim_whitespace=True, header=None, names=colnames)
sa_df = pd.read_csv(fname_sa, delim_whitespace=True, header=None, names=colnames)

"""
 Get info out
"""
dry_peat_vol_no_dams = 40191.730578848255 # normalization value

mc_df = mc_df[mc_df.i != 0]
number_dams = (2,4,6,8,10,12,14,16,18,20)
mean_mc = [mc_df[mc_df.ndams == i]['dry_peat_vol'].mean()/dry_peat_vol_no_dams for i in number_dams]
max_mc = [mc_df[mc_df.ndams == i]['dry_peat_vol'].max()/dry_peat_vol_no_dams for i in number_dams]
min_mc = [mc_df[mc_df.ndams == i]['dry_peat_vol'].min()/dry_peat_vol_no_dams for i in number_dams]

sa_df = sa_df.rename(columns=rename_cols)
sa_df.dry_peat_vol = sa_df.dry_peat_vol/dry_peat_vol_no_dams

"""
 Plot
"""

fig, ax = plt.subplots(1)
ax.plot(number_dams, mean_mc)
ax.fill_between(number_dams, max_mc, min_mc, facecolor='red', alpha=0.5 )
ax.scatter(x=sa_df.ndams.to_numpy(), y=sa_df.dry_peat_vol.to_numpy())

fname_fig = r'results_plot.png'
plt.savefig(fname_fig)

plt.figure()
sns.set(style="whitegrid", palette="pastel", color_codes=True)

# Draw a nested violinplot and split the violins for easier comparison
sns.violinplot(x=mc_df.ndams, y=mc_df.dry_peat_vol/dry_peat_vol_no_dams,  inner="quart")
sns.scatterplot(x='ndams', y='dry_peat_vol', data= sa_df) # Somehow I cannot get this right...
#sns.despine(left=True)