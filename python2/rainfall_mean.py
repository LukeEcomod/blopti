#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 17:57:36 2019

@author: Iñaki
"""

import pandas as pd
import numpy as np
import copy

rainfall_fn = r"/home/txart/Downloads/PKU_Rainfall_Analysis_28May2014revised August2014.xls"

df = pd.read_excel(rainfall_fn)

df_zero = copy.deepcopy(df) # Missing data = 0.0

df_zero.loc[df_zero.JUL == 'TTU'] = 0.0

#df_zero = df_zero * (~ df_zero.isna()) # NaN

dry_days_in_a_row_zero = []
dry_days = 0
for key, value in df_zero['JUL'].iteritems():
    if value < 0.01:
        dry_days = dry_days + 1
    else:
        if dry_days != 0: # only save positive values, i.e., don't save information about consecutive rainy days
            dry_days_in_a_row_zero.append(dry_days)
        dry_days = 0
        
dry_days_in_a_row = []
dry_days = 0
for key, value in df['JUL'].iteritems():
    if value < 0.01:
        dry_days = dry_days + 1
    else:
        if dry_days != 0:
            dry_days_in_a_row.append(dry_days)
        dry_days = 0

print 'if NaN = 0, dry days in a row in JULY from 1994 to 2013 (min, mean, max) = ', (min(dry_days_in_a_row_zero), np.array(dry_days_in_a_row_zero).mean(), max(dry_days_in_a_row_zero))
print 'if NaN != 0, mean dry days in a row in JULY from 1994 to 2013 (min, mean, max) = ', (min(dry_days_in_a_row) ,np.array(dry_days_in_a_row).mean(), max(dry_days_in_a_row))