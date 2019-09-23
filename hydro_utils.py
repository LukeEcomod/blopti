# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 13:16:55 2018

@author: L1817
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.interpolate import InterpolatedUnivariateSpline as interS 
import pandas as pd


def peat_hydrol_properties(x, unit='g/cm3', var='bd', ptype='A'):
    """
    Peat water retention and saturated hydraulic conductivity as a function of bulk density
    Päivänen 1973. Hydraulic conductivity and water retention in peat soils. Acta forestalia fennica 129.
    see bulk density: page 48, fig 19; degree of humification: page 51 fig 21
    Hydraulic conductivity (cm/s) as a function of bulk density(g/cm3), page 18, as a function of degree of humification see page 51 
    input:
        - x peat inputvariable in: db, bulk density or dgree of humification (von Post)  as array \n
        - bulk density unit 'g/cm3' or 'kg/m3' \n
        - var 'db' if input variable is as bulk density, 'H' if as degree of humification (von Post) \n
        - ptype peat type: 'A': all, 'S': sphagnum, 'C': Carex, 'L': wood, list with length of x 
    output: (ThetaS and ThetaR in m3 m-3)
        van Genuchten water retention parameters as array [ThetaS, ThetaR, alpha, n] \n
        hydraulic conductivity (m/s)
    """
    
    #paras is dict variable, parameter estimates are stored in tuples, the model is water content = a0 + a1x + a2x2, where x is
    para={}                                                                     #'bd':bulk density in g/ cm3; 'H': von Post degree of humification
    para['bd'] ={'pF0':(97.95, -79.72, 0.0), 'pF1.5':(20.83, 759.69, -2484.3),
            'pF2': (3.81, 705.13, -2036.2), 'pF3':(9.37, 241.69, -364.6),
            'pF4':(-0.06, 249.8, -519.9), 'pF4.2':(0.0, 174.48, -348.9)}
    para['H'] ={'pF0':(95.17, -1.26, 0.0), 'pF1.5':(46.20, 8.32, -0.54),
            'pF2': (27.03, 8.14, -0.43), 'pF3':(17.59, 3.22, -0.07),
            'pF4':(8.81, 3.03, -0.10), 'pF4.2':(5.8, 2.27, -0.08)}
    
    intp_pF1={}                                                                 # interpolation functions for pF1        
    intp_pF1['bd'] = interp1d([0.04,0.08,0.1,0.2],[63.,84.,86.,80.],fill_value='extrapolate')
    intp_pF1['H'] = interp1d([1.,4.,6.,10.],[75.,84.,86.,80.],fill_value='extrapolate')
    
    #Saturatated hydraulic conductivity parameters
    Kpara ={'bd':{'A':(-2.271, -9.80), 'S':(-2.321, -13.22), 'C':(-1.921, -10.702), 'L':(-1.921, -10.702)}, 
            'H':{'A':(-2.261, -0.205), 'S':(-2.471, -0.253), 'C':(-1.850, -0.278), 'L':(-2.399, -0.124)}}
    
    vg_ini=(0.88,	0.09, 0.03, 1.3)                                              # initial van Genuchten parameters (porosity, residual water content, alfa, n)

    x = np.array(x)
    prs = para[var]; pF1=intp_pF1[var]
    if unit=='kg/m3'and var=='db': x=x/1000.
    if  np.shape(x)[0] >1 and len(ptype)==1:
        ptype=np.repeat(ptype, np.shape(x)[0])        
    vgen = np.zeros((np.size(x),4))
    Ksat = np.zeros((np.size(x)))
    wcont = lambda x, (a0, a1, a2): a0 + a1*x + a2*x**2.
    van_g = lambda pot, *p:   p[1] + (p[0] - p[1]) / (1. + (p[2] * pot) **p[3]) **(1. - 1. / p[3])   
    K = lambda x, (a0, a1): 10.**(a0 + a1*x) / 100.   # to m/s   
    
    potentials =np.array([0.01, 10.,32., 100.,1000.,10000.,15000. ])
    wc = (np.array([wcont(x,prs['pF0']), pF1(x), wcont(x,prs['pF1.5']), wcont(x,prs['pF2']),
               wcont(x,prs['pF3']), wcont(x,prs['pF4']),wcont(x,prs['pF4.2'])]))/100.
        
    for i,s in enumerate(np.transpose(wc)):
        vgen[i],_= curve_fit(van_g,potentials,s, p0=vg_ini)                      # van Genuchten parameters
        
    for i, a, pt in zip(range(len(x)), x, ptype):
        Ksat[i] = K(a, Kpara[var][pt])                                          # hydraulic conductivity (cm/s -> m/s) 
    
    return vgen, Ksat

def wrc(pF, x=None, var=None):
    """
    vanGenuchten-Mualem soil water retention curve\n
    IN:
        pF - dict['ThetaS': ,'ThetaR': ,'alpha':, 'n':,] OR
           - list [ThetaS, ThetaR, alpha, n]
        x  - soil water tension [m H2O = 0.1 kPa]
           - volumetric water content [vol/vol]
        var-'Th' is x=vol. wat. cont.
    OUT:
        res - Theta(Psii) or Psii(Theta)
    NOTE:\n
        sole input 'pF' draws water retention curve and returns 'None'. For drawing give only one pF-parameter set. 
        if several pF-curves are given, x can be scalar or len(x)=len(pF). In former case var is pF(x), in latter var[i]=pf[i,x[i]]
               
    Samuli Launiainen, Luke 2/2016
    """
    if type(pF) is dict: #dict input
        #Ts, Tr, alfa, n =pF['ThetaS'], pF['ThetaR'], pF['alpha'], pF['n']
        Ts=np.array(pF['ThetaS'].values()); Tr=np.array( pF['ThetaR'].values()); alfa=np.array( pF['alpha'].values()); n=np.array( pF['n'].values())
        m= 1.0 -np.divide(1.0,n)
    elif type(pF) is list: #list input
        pF=np.array(pF, ndmin=1) #ndmin=1 needed for indexing to work for 0-dim arrays
        Ts=pF[0]; Tr=pF[1]; alfa=pF[2]; n=pF[3] 
        m=1.0 - np.divide(1.0,n)
    elif type(pF) is np.ndarray:
        Ts, Tr, alfa, n = pF.T[0], pF.T[1], pF.T[2], pF.T[3]
        m=1.0 - np.divide(1.0,n)
    else:
        print 'Unknown type in pF'
        
    def theta_psi(x): #'Theta-->Psi'
        x=np.minimum(x,Ts) 
        x=np.maximum(x,Tr) #checks limits
        s= ((Ts - Tr) / (x - Tr))#**(1/m)
        Psi=-1e-2/ alfa*(s**(1/m)-1)**(1/n) # in m
        return Psi
        
    def psi_theta(x): # 'Psi-->Theta'
        x=100*np.minimum(x,0) #cm
        Th = Tr + (Ts-Tr)/(1+abs(alfa*x)**n)**m
        return Th           
 
    if var is 'Th': y=theta_psi(x) #'Theta-->Psi'           
    else: y=psi_theta(x) # 'Psi-->Theta'          
    return y


def CWTr(nLyrs, z, dz, pF, Ksat, direction='positive'):
    """
    Returns interpolation functions 
        sto=f(gwl)  profile water storage as a function ofground water level
        gwl=f(sto)  ground water level
        tra=f(gwl)  transissivity
    Input:
        nLyrs number of soil layers
        d depth of layer midpoint
        dz layer thickness
        pF van Genuchten water retention parameters: ThetaS, ThetaR, alfa, n
        Ksat saturated hydraulic conductivity in m s-1. K in m/day.
        direction: positive or negative downwards
    """    
    #-------Parameters ---------------------
    z = np.array(z)   
    dz =np.array(dz)
    #--------- Connection between gwl and water storage------------

    if direction =='positive':
        print 'no positive direction available'
        import sys; sys.exit()
        gwl=np.linspace(0.,sum(dz),500)
        sto = [sum(wrc(pF, x = np.minimum(z-g, 0.0))*dz) for g in gwl]     #equilibrium head m                
    else:
        gwl=np.linspace(0.,-sum(dz),100)        
        sto = [sum(wrc(pF, x = np.minimum(z+g, 0.0))*dz) for g in gwl]     #equilibrium head m
        gwlabove = [2.0,1.5,0.5]        
        stoabove = [sto[0]+0.75*gwlabove[0], sto[0]+0.3*gwlabove[1], sto[0]+0.2*gwlabove[2]]
        stoT=list(stoabove)+list(sto)
        gwlT=list(gwlabove)+list(gwl)                
        gwlToSto = interp1d(np.array(gwlT), np.array(stoT), fill_value='extrapolate')
        stoT = list(stoT); gwl= list(gwlT)        
        sto.reverse(); gwl.reverse()
        stoToGwl =interp1d(np.array(stoT), np.array(gwlT), fill_value='extrapolate')
        cc=np.gradient(gwlToSto(gwlT))/np.gradient(gwlT)
        cc[cc<0.2]=0.2
        C = interp1d(np.array(gwlT), cc, bounds_error=False, fill_value=(0.,1.) )  #storage coefficient function   
        #C = UnivariateSpline(np.array(gwlT), cc, s=10)                    
        #C=interS(np.array(gwlT), cc, k=5)

    #import matplotlib.pylab as plt
    #plt.plot(gwlT, cc, 'ro')
    #plt.plot(gwlT, C(gwlT), 'b-')
    #import sys; sys.exit()
    gwlToSto = interp1d(np.array(gwlT), np.array(stoT), fill_value='extrapolate') 
    stoT = list(stoT); gwlT= list(gwlT)        
    stoT.reverse(); gwlT.reverse()
    stoToGwl =interp1d(np.array(stoT), np.array(gwlT), fill_value='extrapolate')

    del gwlT, stoT
        
    #----------Transmissivity-------------------
    K = np.array(Ksat*86400.)   #from m/s to m/day
    tr =[sum(K[t:]*dz[t:]) for t in range(nLyrs)]        
    if direction=='positive':        
        gwlToTra = interS(z, np.array(tr))            
    else:
        z= list(z);  z.reverse(); tr.reverse()
        gwlToTra = interS(-np.array(z), np.array(tr), k=3, ext='const') # returns limiting value outside interpolation domain 
    del tr
    return gwlToSto, stoToGwl, gwlToTra, C

def peat_map_interp_functions():
    """
    OUTPUT: Produces two dictionaries of  functions for each soil type.
        h_to_tra_dict is the interp. func. that maps gwt to transmissivity for the whole saturated depth.
        tr_cut_dict 
    """
    # Soil parameters
    spara ={
    'gen':{'nLyrs':400, 'dzLyr': 0.05}, # General soil parameters, common to all soil types
    
    'Water':{'ref': 1, # reference number that appears on the peat type map
            'vonP top': [1,1,1,1,2,2,2,2,2,3,3,3,3,3,3,4,4,4,4,4,4,5,5,6,6,7,7,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8],
            'vonP bottom': 10, 'Kadjust':20.0,
            'peat type top':'L', 'peat type bottom':['S']},
    
    'Forest':{'ref': 2, # reference number that appears on the peat type map
            'vonP top': [5,5,6,6,7,7,8,8,8,8,8,8,8,8,8,8,8,8,8,8],
            'vonP bottom': 10, 'Kadjust':40.0,
            'peat type top':'L', 'peat type bottom':['S']},
    
    'Secondaryforest-shrub':{'ref': 3, # reference number that appears on the peat type map
            'vonP top': [5,5,6,6,7,7,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8],
            'vonP bottom': 10, 'Kadjust':40.0,
            'peat type top':'L', 'peat type bottom':['S']},
    
    'Plantation':{'ref': 4, # reference number that appears on the peat type map
            'vonP top': [5,5,6,6,7,7,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8],
            'vonP bottom': 10, 'Kadjust':40.0,
            'peat type top':'L', 'peat type bottom':['S']},
            
    'Agriculture':{'ref': 5, # reference number that appears on the peat type map
            'vonP top': [5,5,6,6,7,7,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8],
            'vonP bottom': 10, 'Kadjust':40.0,
            'peat type top':'L', 'peat type bottom':['S']},
             
    'Wetland':{'ref': 6, # reference number that appears on the peat type map
        'vonP top': [5,5,6,6,7,7,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8],
        'vonP bottom': 10, 'Kadjust':40.0,
        'peat type top':'L', 'peat type bottom':['S']},
               
    'Developed':{'ref': 7, # reference number that appears on the peat type map
        'vonP top': [5,5,6,6,7,7,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8],
        'vonP bottom': 10, 'Kadjust':40.0,
        'peat type top':'L', 'peat type bottom':['S']},
    }
    
    # Common to all soil types
    nLyrs = spara['gen']['nLyrs'] # Number of layers
    dz = np.ones(nLyrs)*spara['gen']['dzLyr'] # thickness of layers, m
    z = np.cumsum(dz)-dz/2.  # depth of the layer center point, m
    
    # Loop through all soil types to construct dictionary h_to_tra
    h_to_tra_and_C_dict = {}
    
    for peat_type in [i for i in spara.keys() if i != 'gen']:
        lenvp=len(spara[peat_type]['vonP top'])    
        vonP = np.ones(nLyrs)*spara[peat_type]['vonP bottom']; vonP[0:lenvp] = spara[peat_type]['vonP top']  # degree of  decomposition, von Post scale
        ptype = spara[peat_type]['peat type bottom']*nLyrs
        peat_type_top_list = [spara[peat_type]['peat type top']]*lenvp
        lenpt = len(spara[peat_type]['peat type top']); ptype[0:lenpt] = peat_type_top_list  
        pF, Ksat = peat_hydrol_properties(vonP, var='H', ptype=ptype)  # peat hydraulic properties after Päivänen 1973    
        hToSto, _, hToTra, C = CWTr(nLyrs, z, dz, pF, Ksat*spara[peat_type]['Kadjust'], direction='negative') # interpolated storage, transmissivity and diff water capacity functions

        h_to_tra_and_C_dict[spara[peat_type]['ref']] = {'name': peat_type, 'fullTra': hToTra(0.0), 'hToTra':hToTra, 'hToSto':hToSto, 'C':C}

    return h_to_tra_and_C_dict


def peat_map_h_to_tra(soil_type_mask, gwt, h_to_tra_and_C_dict):
    """
    Input:
        - soil_type_mask: nparray or flattened nparray of dim the DEM, and peat soil type numbers as elements.
        - gwt: nparray or flattened nparray of gwt.
            If gwt = phi-ele in the hydrology code, then the output is the full depth transmissivity
            If gwt = bottom elevation - ele, then the ouput is the transmissivity to be cut from the above full depth trans.
        - h_to_tra_and_C_dict: dict. Output of peat_map_interp_functions().
    
    Output:
        - tra: Flattened nparray of new transmissivities.
    """
    # MAYBE READ SOIL ARRAY TYPE HERE?
    soil_type_mask = np.ravel(soil_type_mask) # in case it is not flattened
    gwt = np.ravel(gwt)
    
    tra = np.ones(np.shape(soil_type_mask))*-999 # Initialize output array with nodata entries
    
    if soil_type_mask.size != gwt.size:
        raise ValueError('The two should have the same dimensions')
        
    for soil_type_number, value in h_to_tra_and_C_dict.iteritems():
        indices = np.where(soil_type_mask == soil_type_number)
        if np.shape(indices)[1]>0:
            tra[indices] = value['hToTra'](gwt[indices])
    
    return tra

def peat_map_h_to_sto(soil_type_mask, gwt, h_to_tra_and_C_dict):
    """
    Input:
        - soil_type_mask: nparray or flattened nparray of dim the DEM, and peat soil type numbers as elements.
        - gwt: nparray or flattened nparray of gwt.
            If gwt = phi-ele in the hydrology code, then the output is the full depth storage coeff
            If gwt = bottom elevation - ele, then the ouput is the storage coeff to be cut from the above full depth trans.
        - h_to_tra_and_C_dict: dict. Output of peat_map_interp_functions().
    
    Output:
        - sto: Flattened nparray of new storage coeffs C.
    """
    # MAYBE READ SOIL ARRAY TYPE HERE?
    soil_type_mask = np.ravel(soil_type_mask) # in case it is not flattened
    gwt = np.ravel(gwt)
    
    sto = np.ones(np.shape(soil_type_mask))*-999 # Initialize output array with nodata entries
    
    if soil_type_mask.size != gwt.size:
        raise ValueError('The two should have the same dimensions')
        
    for soil_type_number, value in h_to_tra_and_C_dict.iteritems():
        indices = np.where(soil_type_mask == soil_type_number)
        if np.shape(indices)[1]>0:
            sto[indices] = value['C'](gwt[indices])
    
    return sto




def getRainfall(rainFile='C:\Users\L1817\Dropbox\PhD\Computation\hydro to Inaki\\rainfall.csv'):
    df=pd.read_csv(rainFile, names=['Date', 'P mm'], skiprows=1)
    df['Date']= pd.to_datetime(df['Date'])
    df.index= df['Date']
    del df['Date']     
    return df