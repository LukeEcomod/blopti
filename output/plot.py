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

fname_mc = r'results_mc_3.txt'
fname_mc_cum = r'results_mc_3_cumulative.txt'
fname_mc_quasi = r"results_mc_quasi_3.txt"
fname_ga = r'results_ga_3.txt'
fname_sa = r'results_sa_3.txt'
fname_summary = r'concise_output_opti.txt'

colnames = ['i', 'dry_peat_vol', 'ndams', 'niter', 'days', 'day_week', 'month', 'day_month', 'time', 'yr', 'water_changed_canals' ]
colnames_sa = ['dry_peat_vol', 'ndams', 'niter', 'days', 'day_week', 'month', 'day_month', 'time', 'yr' ]
colnames_ga = ['dry_peat_vol', 'ndams', 'niter', 'days', 'day_week', 'month', 'day_month', 'time', 'yr' ]
#rename_cols_ga = {'day_month':'dry_peat_vol', 'niter':'ndams', 'c': 'niter', 'b':'days', 'blocks':'day_week', 'day_week':'month', 'days':'day_month', 'month':'time', 'ndams':'yr', 'a': 'b1', 'time':'b2', 'yr':'b3', 'dry_peat_vol':'b4'}

mc_df = pd.read_csv(fname_mc_cum, delim_whitespace=True, header=None, names=colnames, usecols=[0,1,2,3,4,5,6,7,8,9,10])
mc_quasi_df = pd.read_csv(fname_mc_quasi, delim_whitespace=True, header=None, names=colnames, usecols=[0,1,2,3,4,5,6,7,8,9,10])
ga_df = pd.read_csv(fname_ga, delim_whitespace=True, header=None, names=colnames_ga, usecols=[0,1,2,3,4,5,6,7,8])
sa_df = pd.read_csv(fname_sa, delim_whitespace=True, header=None, names=colnames_sa, usecols=[0,1,2,3,4,5,6,7,8])
summary_df = pd.read_csv(fname_summary, delim_whitespace=True)


"""
 Get info out
"""

#dry_peat_vol_no_dams = 40191.730578848255 # normalization value OLD
#dry_peat_vol_no_dams = 7650.495525801664 # normalization value after 3 days. 2 December
dry_peat_vol_no_dams = summary_df['dry_peat_vol_cumulative'][0]
number_dams = (5,10,20, 30, 40, 50, 60, 70, 80)

mc_df = mc_df[mc_df.i != 0]
mc_plot_vol = mc_df[mc_df['days']==3]
mean_mc_plot_vol = [mc_plot_vol[mc_plot_vol.ndams == i]['dry_peat_vol'].mean()/dry_peat_vol_no_dams*100 for i in number_dams]
max_mc_plot_vol = [mc_plot_vol[mc_plot_vol.ndams == i]['dry_peat_vol'].max()/dry_peat_vol_no_dams*100 for i in number_dams]
min_mc_plot_vol = [mc_plot_vol[mc_plot_vol.ndams == i]['dry_peat_vol'].min()/dry_peat_vol_no_dams*100 for i in number_dams]

mean_mcquasi_plot_vol = [mc_quasi_df[mc_quasi_df.ndams == i]['dry_peat_vol'].mean()/dry_peat_vol_no_dams*100 for i in number_dams]
max_mcquasi_plot_vol = [mc_quasi_df[mc_quasi_df.ndams == i]['dry_peat_vol'].max()/dry_peat_vol_no_dams*100 for i in number_dams]
min_mcquasi_plot_vol = [mc_quasi_df[mc_quasi_df.ndams == i]['dry_peat_vol'].min()/dry_peat_vol_no_dams*100 for i in number_dams]


sa_ndays = sa_df[sa_df['ndams']==20]

sa_df.dry_peat_vol = sa_df.dry_peat_vol/dry_peat_vol_no_dams*100

ga_df.dry_peat_vol = ga_df.dry_peat_vol/dry_peat_vol_no_dams*100


# Choose to plot ndams from number_dams above
sa_plot = summary_df[summary_df['mode'] == 'sa_fullopti']

ga_plot = summary_df[summary_df['mode'] == 'ga_fullopti']


# CWL and Volume dry peat
#ga_fullopti_cwl_vdp = np.array([[100.17000408172562, 206.5699918508545, 294.7300499916108, 413.46002569199203, 490.95001428128023, 611.140032672881, 630.9599639892541, 655.8200357675498, 741.6799679756032],
#                   [99.42081341, 98.86536757, 98.27517161, 97.69790191, 97.28177471,96.68626366, 96.60555668, 96.22210773, 95.83706336]])
#sa_fullopti_cwl_vdp = np.array([[88.14001469612094, 154.89002861976618, 239.2000149965306, 335.2400191068692, 335.7899886608159, 510.699984335907, 473.95000741482465, 545.1500374794063, 583.9700692176839],
#                   [99.48070425, 99.19026325, 98.81467022, 98.32784208, 98.11629296,97.55972738, 97.50128656, 97.10690771, 96.84186172]])
#    
#ga_simpleopti_cwl_vdp = np.array([[169.89998197, 289.469981575, 486.719960117, 632.6099651813, 742.240028381, 846.459938741, 931.629919863, 1040.68997881, 1085.7399986],
#                                  [99.429750557, 99.028091737,  98.26186190, 97.71969008691086, 97.47046337112018, 96.8566965286157, 96.625790926368, 95.8517598466997, 95.649595692221]])
#rule_based_cwl_vdp = np.array([[24.620006227493246, 41.410006046295],
#                               [99.89676051674989, 99.77789666106682]])

ga_fullopti_cwl_vdp = np.array([np.array(summary_df[summary_df['mode'] == 'ga_fullopti'].CWLchange),
                                  np.array(summary_df[summary_df['mode'] == 'ga_fullopti'].dry_peat_vol_cumulative)])
sa_fullopti_cwl_vdp = np.array([np.array(summary_df[summary_df['mode'] == 'sa_fullopti'].CWLchange),
                                  np.array(summary_df[summary_df['mode'] == 'sa_fullopti'].dry_peat_vol_cumulative)])
ga_simpleopti_cwl_vdp = np.array([np.array(summary_df[summary_df['mode'] == 'ga_simpleopti'].CWLchange),
                                  np.array(summary_df[summary_df['mode'] == 'ga_simpleopti'].dry_peat_vol_cumulative)])
rule_based_cwl_vdp = np.array([np.array(summary_df[summary_df['mode'] == 'rule_based'].CWLchange),
                                  np.array(summary_df[summary_df['mode'] == 'rule_based'].dry_peat_vol_cumulative)])

# For tomorrow: get this data rom summary_df by: 
# summary_df[summary_df['mode'] == 'ga_simpleopti'] etc.
"""
 Plot V_dry_peat vs ndams
"""

fig, ax = plt.subplots(1)
ax.fill_between(number_dams, max_mc_plot_vol, min_mc_plot_vol, facecolor='wheat', alpha=0.5, label='random range')
ax.scatter(x=number_dams, y=mean_mc_plot_vol, alpha=0.8, color='darkorange', label='random mean', marker='P')
#ax.plot(number_dams, mean_mcquasi_plot_vol, alpha=1.0, color='purple', label='quasi-random mean')
#ax.fill_between(number_dams, max_mcquasi_plot_vol, min_mcquasi_plot_vol, facecolor='purple', alpha=0.4, label='quasi-random range')
ax.set_xlabel('Number of blocks')
ax.set_ylabel('Volume fraction of dry peat (%)')

ax.scatter(x=sa_plot.ndams.to_numpy(), y=sa_plot.dry_peat_vol_cumulative.to_numpy(), label='SA', alpha=0.5, color='blue', marker='s')
ax.scatter(x=ga_plot.ndams.to_numpy(), y=ga_plot.dry_peat_vol_cumulative.to_numpy(), label='GA', alpha=0.7, color='red')
ax.scatter(x=number_dams[:len(ga_simpleopti_cwl_vdp[1])], y=ga_simpleopti_cwl_vdp[1], color='green', alpha=0.8, marker='x', label='SO')
ax.scatter(x=number_dams[:len(rule_based_cwl_vdp[1])], y=rule_based_cwl_vdp[1], color='black', alpha=0.8, label='rule-based')
plt.legend()

#fname_fig = r'results_plot.png'
#plt.savefig(fname_fig)


#sns.set(style="whitegrid", palette="pastel", color_codes=True)
#sns.catplot(x='ndams', y='dry_peat_vol', kind='violin', color='red', data=mc_plot_vol)
#sns.despine(left=True)

"""
Plot relative improvement
"""
mmc_improvement = np.sort(100. - np.array(mean_mc_plot_vol))
sa_improvement = 100. - np.sort(sa_plot.dry_peat_vol_cumulative)[::-1]
ga_improvement = 100. - np.sort(ga_plot.dry_peat_vol_cumulative)[::-1]
simple_improvement = 100. -np.sort(ga_simpleopti_cwl_vdp[1])[::-1]

fig, ax = plt.subplots(1)

ax.set_xlabel('Number of blocks')
ax.set_ylabel('Relative improvement wrt random mean')

ax.scatter(x=number_dams, y=sa_improvement/mmc_improvement, label='SA', alpha=0.5, color='blue', marker='s')
ax.scatter(x=number_dams, y=ga_improvement/mmc_improvement, label='GA', alpha=0.7, color='red')
ax.scatter(x=number_dams, y=simple_improvement/mmc_improvement, color='green', alpha=0.8, marker='x', label='SO')
plt.grid(True, which='major', axis='y')
#plt.legend()


"""
Plot MB
"""
mb_dams = list(number_dams)
mb_dams.append(0); mb_dams.sort()
sa_vdry = np.sort(sa_plot.dry_peat_vol_cumulative)[::-1]
ga_vdry = np.sort(ga_plot.dry_peat_vol_cumulative)[::-1]
simple_vdry = np.sort(ga_simpleopti_cwl_vdp[1])[::-1]
best_opti = [100.0]

for i in range(0,len(mb_dams)-1):
    opti_pool = [sa_vdry[i], ga_vdry[i], simple_vdry[i]]
    best_opti.append( min(opti_pool))

mean_rand =[100.0] + mean_mc_plot_vol
mb_opti = []; mb_random = []
for i in range(0,len(mb_dams)-1):
    margben_opti = (-best_opti[i+1] + best_opti[i]) / (mb_dams[i+1] - mb_dams[i])
    margben_random = (-mean_rand[i+1] + mean_rand[i]) / (mb_dams[i+1] - mb_dams[i])
    mb_opti.append(margben_opti)
    mb_random.append(margben_random)
    
mb_ga = mb_opti[0:-2] # separation for the plot
mb_sopti = mb_opti[-2:]

fig, ax = plt.subplots(1)
ax.set_xlabel('d')
ax.set_ylabel('MB(d) %')
ax.scatter(x= mb_dams[0:-3], y=mb_ga, label='GA', alpha=0.7, color='red')
ax.scatter(x= mb_dams[-3:-1], y=mb_sopti, color='green', alpha=0.8, marker='x', label='SO')
ax.scatter(x= mb_dams[:-1], y=mb_random, alpha=0.8, color='darkorange', label='random mean', marker='P')


"""
Plot dry vs ndays
"""
mc_df2 = mc_df[mc_df['niter'] == 2000]
mc_df_ndays = mc_df2[mc_df2['ndams'] == 20]


days = (3, 5, 10, 15, 20)
# dpv no dams, updated 12Dec
dpv = {3: 8255.131496485912, 5: 10425.012343947397, 10: 14688.078154032499, 15:17975.698276311792, 20: 20709.290669916085} # normalization value after x days.

mc_mean_dpv = []; mc_min_dpv = []; mc_max_dpv = []
sa_days = []
for day in sorted(dpv.iterkeys()):
    mc_mean_dpv.append( mc_df_ndays[mc_df_ndays.days == day]['dry_peat_vol'].mean()/dpv[day]*100 )
    mc_max_dpv.append( mc_df_ndays[mc_df_ndays.days == day]['dry_peat_vol'].max()/dpv[day]*100 )
    mc_min_dpv.append( mc_df_ndays[mc_df_ndays.days == day]['dry_peat_vol'].min()/dpv[day]*100 )
    if day != 15: # Missing the 15 day data for now
        sa_days.append(sa_ndays[sa_ndays.days==day].dry_peat_vol.to_numpy()[0]/dpv[day]*100)
    
    
#plt.figure()
#sns.set(style="whitegrid", palette="pastel", color_codes=True)
#sns.boxplot(x=mc_df2.days, y=mc_df2.dry_peat_vol/dpv[3])  
    
fig, ax = plt.subplots(1)
ax.plot(days, mc_mean_dpv, alpha=0.8, color='darkorange', label='random mean', marker='P')
ax.fill_between(days, mc_max_dpv, mc_min_dpv, facecolor='wheat', alpha=0.5, label='random range')
ax.scatter([3,5,10,20], sa_days, label='SA', alpha=0.5, color='blue', marker='s')
ax.set_xlabel('Number of dry days')
ax.set_ylabel('Volume fraction of dry peat (%)')
ax.set_title('20 blocks')
plt.legend()

#ax.scatter(x=sa_plot.ndams.to_numpy(), y=sa_plot.dry_peat_vol.to_numpy(), label='SA', alpha=0.8, color='orange')


"""
Plot correlation Vdry peat vs CWL change
"""

mc_df3 = mc_df2[mc_df2['days'] == 3]


fig, ax = plt.subplots(1)
ax.scatter(x=mc_df3.water_changed_canals, y=mc_df3.dry_peat_vol/dry_peat_vol_no_dams*100, s=1.5, alpha=0.1, color='darkorange', label='random mean', marker='P')

ax.scatter(x=sa_fullopti_cwl_vdp[0], y=sa_fullopti_cwl_vdp[1], label='SA', alpha=0.5, color='blue', marker='s')
i=0
for x, y in zip(sa_fullopti_cwl_vdp[0], sa_fullopti_cwl_vdp[1]):
    plt.text(x, y, number_dams[i], color="blue", fontsize=12)
    i = i+1
    
ax.scatter(x=ga_fullopti_cwl_vdp[0], y=ga_fullopti_cwl_vdp[1], label='GA', alpha=0.7, color='red')
i=0
for x, y in zip(ga_fullopti_cwl_vdp[0], ga_fullopti_cwl_vdp[1]):
    plt.text(x, y, number_dams[i], color="red", fontsize=12)
    i = i+1
    
ax.scatter(x=ga_simpleopti_cwl_vdp[0], y=ga_simpleopti_cwl_vdp[1], color='green', alpha=0.8, marker='x', label='SO')
i=0
for x, y in zip(ga_simpleopti_cwl_vdp[0], ga_simpleopti_cwl_vdp[1]):
    plt.text(x, y, number_dams[i], color="green", fontsize=12)
    i = i+1

ax.set_xlabel('CWL change')
ax.set_ylabel('Volume fraction of dry peat (%)')

# Plot legend.
lgnd = plt.legend(loc="lower left", scatterpoints=1, fontsize=10, framealpha=1.0)
lgnd.legendHandles[0]._sizes = [30]
lgnd.legendHandles[1]._sizes = [30]
lgnd.legendHandles[0].set_alpha(1)


# Separated by number of blocks
fig, axes = plt.subplots(3,3)
axes = axes.flatten()
for i, n_dams in enumerate(number_dams):
    mc_reduced = mc_df3[ mc_df3['ndams'] == n_dams]
    axes[i].scatter(x=mc_reduced.water_changed_canals, y=mc_reduced.dry_peat_vol/dry_peat_vol_no_dams*100, s=1.5, alpha=0.5, color='darkorange', label='random mean', marker='P')
    axes[i].scatter(x=ga_fullopti_cwl_vdp[0][i], y=ga_fullopti_cwl_vdp[1][i], label='GA', alpha=0.7, color='red')
    axes[i].scatter(x=sa_fullopti_cwl_vdp[0][i], y=sa_fullopti_cwl_vdp[1][i], label='SA', alpha=0.5, color='blue', marker='s')

    if i < len(ga_simpleopti_cwl_vdp[0]):
        axes[i].scatter(x=ga_simpleopti_cwl_vdp[0][i], y=ga_simpleopti_cwl_vdp[1][i], color='green', alpha=0.8, marker='x', label='SO')

    axes[i].set_title(str(n_dams) + 'blocks')
    axes[i].set_xlabel('CWL change')
    axes[i].set_ylabel('Volume fraction of dry peat (%)')

# legends for subplot
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center')

# only outer axes labels
for ax in axes.flat:
    ax.label_outer()




plt.show()
