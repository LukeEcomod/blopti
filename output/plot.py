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

colnames = ['i', 'dry_peat_vol', 'ndams', 'niter', 'days', 'day_week', 'month', 'day_month', 'time', 'yr', 'water_changed_canals' ]
rename_cols_sa = {'i':'dry_peat_vol', 'dry_peat_vol':'ndams', 'ndams':'niter', 'niter':'days', 'days':'day_week', 'day_week':'month', 'month':'day_month', 'day_month':'time', 'time':'yr' }
colnames_ga = {'dry_peat_vol', 'ndams', 'niter', 'days', 'day_week', 'month', 'day_month', 'time', 'yr', 'blocks', 'a', 'b', 'c'}
rename_cols_ga = {'day_month':'dry_peat_vol', 'niter':'ndams', 'c': 'niter', 'b':'days', 'blocks':'day_week', 'day_week':'month', 'days':'day_month', 'month':'time', 'ndams':'yr', 'a': 'b1', 'time':'b2', 'yr':'b3', 'dry_peat_vol':'b4'}

mc_df = pd.read_csv(fname_mc, delim_whitespace=True, header=None, names=colnames)
ga_df = pd.read_csv(fname_ga, delim_whitespace=True, header=None, names=colnames_ga)
sa_df = pd.read_csv(fname_sa, delim_whitespace=True, header=None, names=colnames)

"""
 Get info out
"""

#dry_peat_vol_no_dams = 40191.730578848255 # normalization value OLD
dry_peat_vol_no_dams = 7650.495525801664 # normalization value after 3 days. 2 December
number_dams = (5,10,20, 30, 40, 50, 60, 70, 80)

mc_df = mc_df[mc_df.i != 0]
mc_plot_vol = mc_df[mc_df['niter']==2000]
mc_plot_vol = mc_plot_vol[mc_plot_vol['days']==3]
mean_mc_plot_vol = [mc_plot_vol[mc_plot_vol.ndams == i]['dry_peat_vol'].mean()/dry_peat_vol_no_dams*100 for i in number_dams]
max_mc_plot_vol = [mc_plot_vol[mc_plot_vol.ndams == i]['dry_peat_vol'].max()/dry_peat_vol_no_dams*100 for i in number_dams]
min_mc_plot_vol = [mc_plot_vol[mc_plot_vol.ndams == i]['dry_peat_vol'].min()/dry_peat_vol_no_dams*100 for i in number_dams]

sa_df = sa_df.rename(columns=rename_cols_sa)
sa_df.dry_peat_vol = sa_df.dry_peat_vol/dry_peat_vol_no_dams*100

ga_df = ga_df.rename(columns=rename_cols_ga)
ga_df.dry_peat_vol = ga_df.dry_peat_vol/dry_peat_vol_no_dams*100

# slice out
ga_df = ga_df[21:]
sa_df = sa_df[sa_df['dry_peat_vol'] < 100]


# Choose to plot ndams from number_dams above
sa_plot = sa_df.loc[sa_df['ndams'].isin(number_dams)]
ga_plot = ga_df.loc[ga_df['ndams'].isin(number_dams)]


sa_plot = sa_plot[sa_plot['day_month']==30]


"""
 Plot V_dry_peat vs ndams
"""

fig, ax = plt.subplots(1)
ax.plot(number_dams, mean_mc_plot_vol, alpha=1.0, color='red', label='random mean')
ax.fill_between(number_dams, max_mc_plot_vol, min_mc_plot_vol, facecolor='pink', alpha=0.7, label='random range')
ax.set_xlabel('Number of dams')
ax.set_ylabel('Volume fraction of dry peat (%)')

ax.scatter(x=sa_plot.ndams.to_numpy(), y=sa_plot.dry_peat_vol.to_numpy(), label='SA', alpha=0.8, color='orange')
ax.scatter(x=ga_plot.ndams.to_numpy(), y=ga_plot.dry_peat_vol.to_numpy(), label='GA', alpha=0.5, color='blue')
plt.legend()

#fname_fig = r'results_plot.png'
#plt.savefig(fname_fig)


#sns.set(style="whitegrid", palette="pastel", color_codes=True)

# Draw a nested violinplot and split the violins for easier comparison
#sns.violinplot(x=mc_df.ndams, y=mc_df.dry_peat_vol/dry_peat_vol_no_dams,  inner="quart")

#sns.despine(left=True)
"""
Plot relative improvement
"""
mmc_improvement = np.sort(100. - np.array(mean_mc_plot_vol))
sa_improvement = 100. - np.sort(sa_plot.dry_peat_vol)[::-1]
ga_improvement = 100. - np.sort(ga_plot.dry_peat_vol)[::-1]

fig, ax = plt.subplots(1)

ax.set_xlabel('Number of dams')
ax.set_ylabel('Relative improvement wrt random mean')

ax.scatter(x=number_dams, y=sa_improvement/mmc_improvement, label='SA', alpha=0.8, color='orange')
ax.scatter(x=number_dams, y=ga_improvement/mmc_improvement, label='GA', alpha=0.5, color='blue')
plt.legend()



"""
Plot dry vs ndays
"""
mc_df2 = mc_df[mc_df['niter'] == 2000]
days = (3, 5, 10, 15, 20)
dpv = {3: 7650.495525801664, 5: 9457.085131423835, 10: 12914.799397751349, 15:15441.81142738325, 20: 17467.949267817334} # normalization value after x days. 3 December

mc_mean_dpv = []; mc_min_dpv = []; mc_max_dpv = []
for day in sorted(dpv.iterkeys()):
    mc_mean_dpv.append( mc_df2[mc_df2.days == day]['dry_peat_vol'].mean()/dpv[3]*100 )
    mc_max_dpv.append( mc_df2[mc_df2.days == day]['dry_peat_vol'].max()/dpv[3]*100 )
    mc_min_dpv.append( mc_df2[mc_df2.days == day]['dry_peat_vol'].min()/dpv[3]*100 )
    
#plt.figure()
#sns.set(style="whitegrid", palette="pastel", color_codes=True)
#sns.boxplot(x=mc_df2.days, y=mc_df2.dry_peat_vol/dpv[3])  
    
fig, ax = plt.subplots(1)
ax.plot(days, mc_mean_dpv, alpha=1.0, color='red', label='random mean')
ax.fill_between(days, mc_max_dpv, mc_min_dpv, facecolor='pink', alpha=0.7, label='random range')
ax.set_xlabel('Number of dry days')
ax.set_ylabel('Volume fraction of dry peat (%)')

#ax.scatter(x=sa_plot.ndams.to_numpy(), y=sa_plot.dry_peat_vol.to_numpy(), label='SA', alpha=0.8, color='orange')


"""
Plot correlation Vdry peat vs CWL change
"""
mc_df3 = mc_df2[mc_df2['days'] == 3]
fig, ax = plt.subplots(1)
ax.scatter(x=-mc_df3.water_changed_canals, y=mc_df3.dry_peat_vol)
ax.set_xlabel('CWL change')
ax.set_ylabel('Volume fraction of dry peat (%)')


plt.show()
