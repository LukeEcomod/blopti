# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 14:13:10 2018

@author: L1817
"""
import pandas as pd
import numpy as np


def getParams(ParamFile, shname , lrs):
    #-----Profile parameters --------------
    #self.nLyrs = 40 
    profPara={'nLyrs':lrs,  'profname':'testi' }
    nLyrs = profPara['nLyrs']        
    #------Parameters from file----------
    pFdf = pd.read_excel(ParamFile, sheet_name='Retention', skiprows=0)          #get database
    pFdf=pFdf.set_index('Number')                                               #index by soil number
    allLyrs =pd.read_excel(ParamFile, sheet_name=shname, skiprows=0)          #get the soil codes in the profie
    dz = allLyrs['dz']; lyrs = allLyrs['Lyrs']       
    pF = pFdf.loc[list(lyrs)].copy()                                             #filter with the list
    pF['lyr']=range(len(pF))                                                    #add layer number to the df
    pF = pF.set_index('lyr')                                                    #set it to index
    pF['dz']=dz                                                                 #thickness of layer (m)  
    z = np.cumsum(dz) - dz/2.0                                                  #depth of node center (m)     
    pF['z']= z
    pF = pF[:nLyrs].to_dict()                                                   #cut the unnecessary layer away
    d={}
    d['profPara']=profPara; d['pF']=pF;     
    print 'soil para from sheet' , shname   
    return d      



def ReadInput(filename):
    dfparams = pd.read_excel(filename, sheet_name='Gen_Params') 
    
    n_nodes = int(dfparams['n_nodes'][0])
    peat_height = float(dfparams['peat_height'][0])
    height = float(dfparams['height'][0])
    block_height = float(dfparams['block_height'][0])
    n_canals = int(dfparams['n_canals'][0])
    n_blocks = int(dfparams['n_blocks'][0])
    nodes_per_canal = list(dfparams['npc_' + str(n_nodes)].dropna())
    
    dfinput = pd.read_excel(filename, sheet_name='Input')
    originalWTcanal = list(dfinput['wtlcan'])
    srfccanal = list(dfinput['srfcan'])    

    return n_nodes, peat_height, height, block_height, n_blocks, n_canals, nodes_per_canal, originalWTcanal, srfccanal

def read_precipitation():
    """
    Reads Pekanbaru airport 2012 weather data.
    Returns numpy array with 2012 (1 year) daily values 
    """
#    rainfall_fn = r"C:\Users\03125327\github\dd_winrock\data\2012_rainfall.xlsx" #Luke
    rainfall_fn = r"/home/txart/Programming/GitHub/dd_winrock/data/2012_rainfall.xlsx"
    df = pd.read_excel(rainfall_fn, names=['', 'RAW_DATA','Fill_nodata','','','','',''])
    return df['Fill_nodata'].to_numpy()
    
    

