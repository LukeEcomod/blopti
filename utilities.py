# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 14:22:00 2018

@author: L1817
"""

import numpy as np
import copy
import random
import scipy.signal



def peel_raster(raster, catchment_mask):
    """
    Given a raster and a mask, gets the "peeling" or "shell" of the raster. (Peeling here are points within the raster)
    Input:
        - raster: 2dimensional nparray. Raster to be peeled. The peeling is part of the raster.
        - catchment_mask: 2dim nparray of same size as raster. This is the fruit in the peeling.
    Output:
        - peeling_mask: boolean nparray. Tells where the peeling is.

    """
    # catchment mask boundaries by convolution
    conv_double = np.array([[0,1,1,1,0],
                            [1,1,1,1,1],
                            [1,1,0,1,1],
                            [1,1,1,1,1],
                            [0,1,1,1,0]])
    bound_double = scipy.signal.convolve2d(catchment_mask, conv_double, boundary='fill', fillvalue=False)
    peeling_mask = np.ones(shape=catchment_mask.shape, dtype=bool)
    peeling_mask[bound_double[2:-2,2:-2]==0] = False; peeling_mask[bound_double[2:-2,2:-2]==20] = False
    
    peeling_mask = (catchment_mask*peeling_mask) > 0

    return peeling_mask


# NEW 23.11. ONE STEP OR MOVEMENT IN THE SIMULATED ANNEALING.
def switch_one_dam(oWTcanals, surface_canals, currentWTcanals, block_height, dams_location, n_canals, CNM):
    """
        Randomly chooses which damn to take out and where to put it again.
    Computes which are "prohibited nodes", where the original water level of the canal is
    lower than the current (i.e., has been affected by the placement of a dam), before making the
    otherwise random decision of locating the dam.
    
    OUTPUT
        - new_wt_canal: wl in canal after removing old and placing a new dam.
    """
        
    # Select dam to add
    # Check for prohibited nodes. Minor problem: those points affected by the dam which will be soon removed are also prohibited
    prohibited_node_list = [i for i,_ in enumerate(oWTcanals) if oWTcanals[i] < currentWTcanals[i]]
    candidate_node_list = [e for e in range(0, n_canals) if e not in prohibited_node_list]
    random.shuffle(candidate_node_list) # Happens in-place.
    dam_to_add = candidate_node_list[0]
    
    # Select dam to remove
    random.shuffle(dams_location) # Shuffle to select which canal to remove. Happens in place.
    dam_to_remove = dams_location[0]
    
    dams_location.remove(dam_to_remove)
    dams_location.append(dam_to_add)
    
    # Compute new wt in canals with this configuration of dams
    new_wt_canal = place_dams(oWTcanals, surface_canals, block_height, dams_location, CNM)
    
    return new_wt_canal
    
    
    
    
    
    

def PeatV_weight_calc(canal_mask):
    """ Computes weights (associated to canal mask) needed for peat volume compt.
    
    input: canal_mask -- np.array of dim (nx,ny). 0s where canals or outside catchment, 1s in the rest.
    
    
    output: np.array of dim (nx,ny) with weights to compute energy of sim anneal
    """
    
    xdim = canal_mask.shape[0]
    ydim = canal_mask.shape[1] 
    
    # Auxiliary array of ones and zeros.
    arr_of_ones = np.zeros((xdim+2,ydim+2)) # Extra rows and cols of zeros to deal with boundary cases
    arr_of_ones[1:-1,1:-1] = canal_mask # mask
    
    # Weights array
    weights = np.ones((xdim,ydim)) 
    
    # Returns number of non-zero 0th order nearest neighbours
    def nn_squares_sum(arr, row, i):
        nsquares = 0
        if ((arr[row,i] + arr[row-1,i] + arr[row-1,i-1] + arr[row,i-1]) == 4):
            nsquares += 1
        if ((arr[row,i] + arr[row,i-1] + arr[row+1,i-1] + arr[row+1,i]) == 4):
            nsquares += 1
        if ((arr[row,i] + arr[row+1,i] + arr[row+1,i+1] + arr[row,i+1]) == 4):
            nsquares += 1
        if ((arr[row,i] + arr[row,i+1] + arr[row-1,i+1] + arr[row-1,i]) == 4):
            nsquares += 1
        return nsquares
    
    
    for j, row in enumerate(arr_of_ones[1:-1, 1:-1]):
        for i, _ in enumerate(row):
            weights[j,i] = nn_squares_sum(arr_of_ones, j+1, i+1)
    
    return weights

def PeatVolume(weights, Z):
    """Computation of dry peat volume. Energy for the simulated annealing.
    INPUT:
        - weights: weights as computed from nn_squares_sum function
        - Z: array of with values = surface dem elevation - wt
    OUTPUT:
        - Dry peat volume. Units?
    """
    
    # This is good code for the future.
#    sur = np.multiply(surface, weights) # element-wise multiplication
#    gwt = np.multiply(gwt, weights) # element-wise multiplication
#    gehi_sur = np.sum(sur) # element-wise sum
#    gehi_gwt = np.sum(gwt)
    zet = np.multiply(Z, weights) # all operations linear, so same as doing them with Z=surface-gwt
    z_sum = np.sum(zet)
    
    #Carefull with gwt over surface!
#    dry_peat_volume = .25 * (ygrid[1]-ygrid[0])*(xgrid[1]-xgrid[0]) * (gehi_sur - gehi_gwt)

    dry_peat_volume = .25 * z_sum
    
    return dry_peat_volume

def print_time_in_mins(time):
    if time >60 and time <3600:
        print "Time spent: ", time/60.0, "minutes"
    elif time >3600:
        print "Time spent: ", time/60.0, "hours"
    else:
        print "Time spent: ", time, "seconds"
        
   
def place_dams(originalWT, srfc, block_height, dams_to_add, CNM):
    """ Takes original water level in canals and list of nodes where to put blocks. Returns updated water level in canals.
    
    Input:
        - originalWT: list. Original water level in canals.
        - srfc: list. DEM value at canals.
        - block_height: float. Determines the new value of the water level as: new_value = surface[add_can] - block_height.
        - dams_to_add: list of ints. positions of dam to add.
        - CNM: propagation or canal adjacency (sparse) matrix.
        
    Output:
        - wt: list. Updated water level in canals.
    """
    
    def addDam(wt, surface, block_height, add_dam, CNM): 
        """ Gets a single canal label and returns the updated wt corresponding to building a dam in that canal
        """
        add_height = surface[add_dam] - block_height
        list_of_canals_to_add = [add_dam]
        
        while len(list_of_canals_to_add) > 0:
            list_of_canals_to_add = list(list_of_canals_to_add) # try to avoid numpyfication
            add_can = list_of_canals_to_add[0]
            
            if wt[add_can] < add_height: # condition for update
                wt[add_can] = add_height
                canals_prop_to = CNM[add_can].nonzero()[1].tolist() # look for propagation in matrix
                list_of_canals_to_add = list_of_canals_to_add + canals_prop_to # append canals to propagate to. If none, it appends nothing.
            
            list_of_canals_to_add = list_of_canals_to_add[1:] # remove add_can from list
            
        return wt
    
    wt = copy.deepcopy(originalWT) # Always start with original wt.
      
    if type(dams_to_add) != list:
        print "WT_update: canals_to_add should have type list"
    
    for add_can in dams_to_add:
        wt = addDam(wt, srfc, block_height, add_can, CNM)
    
    return wt


