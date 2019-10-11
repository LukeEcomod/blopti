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
"""
 Read
"""

fname_mc = r'results_mc.txt'
fname_ga = r'results_ga.txt'
fname_sa = r'results_sa.txt'

colnames = ['i', 'dry_peat_vol', 'ndams', 'niter', 'days', 'day_week', 'month', 'day_month', 'time', 'yr' ]

mc_df = pd.read_csv(fname_mc, delim_whitespace=True, header=None, names=colnames)
ga_df = pd.read_csv(fname_ga, delim_whitespace=True, header=None, names=colnames)
sa_df = pd.read_csv(fname_sa, delim_whitespace=True, header=None, names=colnames)

"""
 Get info out
"""
mc_df = mc_df[mc_df.i != 0]
number_dams = (2,4,6,8,10,12,14,16,18,20)
mean_mc = [mc_df[mc_df.ndams == i]['dry_peat_vol'].mean() for i in number_dams]
max_mc = [mc_df[mc_df.ndams == i]['dry_peat_vol'].max() for i in number_dams]
min_mc = [mc_df[mc_df.ndams == i]['dry_peat_vol'].min() for i in number_dams]

value_sa = 29927.639433454486

"""
 Plot
"""

fig, ax = plt.subplots(1)
ax.plot(number_dams, mean_mc)
ax.fill_between(number_dams, max_mc, min_mc, facecolor='red', alpha=0.5 )
ax.scatter(x=10, y=value_sa)

fname_fig = r'results_plot.png'
plt.savefig(fname_fig)


plt.figure()
sns.set(style="whitegrid", palette="pastel", color_codes=True)

# Draw a nested violinplot and split the violins for easier comparison
sns.violinplot(x=mc_df.ndams, y=mc_df.dry_peat_vol,  inner="quart")
sns.despine(left=True)