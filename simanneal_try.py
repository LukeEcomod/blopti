# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 11:13:41 2018

@author: L1817
"""
from __future__ import print_function
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import preprocess_data,  utilities, hydro
import random
from simanneal import Annealer

time0 = time.time()

class DryPeatVolumeMinimization(Annealer):
    """
    - The variable state = list of blocked canals
    - Moving the state amounts to switching where to put one of the dams and computing the wl in the canals.
    - Energy = amount of dry peat volume.
    - The energy function computes the amount of dry peat from a given state.
    """
    # pass extra data (the distance matrix) into the constructor
    def __init__(self, state, wt_canals):
        self.wt_canals = wt_canals
        super(DryPeatVolumeMinimization, self).__init__(state)  # important!

    def move(self):
        """
        ANALOGOUS TO utilities.switch_one_dam function.
        Randomly chooses which damn to take out and where to put it again.
        Computes which are "prohibited nodes", where the original water level of the canal is
        lower than the current (i.e., has been affected by the placement of a dam), before making the
        otherwise random decision of locating the dam.
        
        OUTPUT
            - updates the self.state variable by giving new locations of dams
        """        
        # Select dam to add
        # Check for prohibited nodes. Minor problem: those points affected by the dam which will be soon removed are also prohibited
        prohibited_node_list = [i for i,_ in enumerate(oWTcanals) if oWTcanals[i] < self.wt_canals[i]]
        candidate_node_list = [e for e in range(0, n_canals) if e not in prohibited_node_list]
        random.shuffle(candidate_node_list) # Happens in-place.
        dam_to_add = candidate_node_list[0]
        
        # Select dam to remove
        random.shuffle(self.state) # Shuffle to select which canal to remove. Happens in place.
        dam_to_remove = self.state[0]
        
        self.state.remove(dam_to_remove)
        self.state.append(dam_to_add)

    def energy(self):
        self.wt_canals = utilities.place_dams(oWTcanals, surface_canals, block_height,
                                self.state, CNM)
        
        wt_canal_arr = np.zeros((ny,nx)) # (nx,ny) array with wt canal height in corresponding nodes
        for canaln, coords in enumerate(c_to_r_list):
            wt_canal_arr[coords] = self.wt_canals[canaln]
            Hinitial[coords] = self.wt_canals[canaln]
        
        e = hydro.hydrology(nx, ny, dx, dy, dt, ele, Hinitial, catchment_mask, wt_canal_arr,
                            value_for_masked= 0.9, diri_bc=None, neumann_bc = 0.0, plotOpt=False)
        return e

        
"""
Read and preprocess data
"""
retrieve_canalarr_from_pickled = False

preprocessed_datafolder= r"C:\Users\L1817\Dropbox\PhD\Computation\Indonesia_WaterTable\ForestCarbon\preprocessed_data"
rst_dem_fn = preprocessed_datafolder + r'\50x50dem_filled_and_interpolated.tif'

if 'cm' and 'cr' and 'c_to_r_list' not in globals():
    datafolder = r"C:\Users\L1817\Dropbox\PhD\Computation\Indonesia_WaterTable\ForestCarbon\data"
    shp_fn = datafolder + r"\canal2017_clip.shp"
    rst_out_fn = preprocessed_datafolder + r'\canals_rasterized.tif'

    cm, cr, c_to_r_list =  preprocess_data.gen_can_matrix_and_raster_from_raster(shp_fn=shp_fn,
                                    rst_dem_fn=rst_dem_fn, rst_out_fn=rst_out_fn,
                                    preprocessed_datafolder=preprocessed_datafolder)

elif retrieve_canalarr_from_pickled==True:
    pickle_folder = r"C:\Users\L1817\ForestCarbon"
    pickled_canal_array_fn = r'\50x50_DEM_and_canals.pkl'
    with open(pickle_folder + pickled_canal_array_fn) as f:
        cm, cr, c_to_r_list = pickle.load(f)
    print ("Canal adjacency matrix and raster loaded from pickled.")
    
else:
    print ("Canal adjacency matrix and raster loaded from memory.")

dem = utilities.read_DEM(rst_dem_fn)
print("DEM read from file")

surface_canals =[dem[coords] for coords in c_to_r_list]

n_canals = len(c_to_r_list)
n_blocks = 3
block_height = 0.4 # water level of canal after placing dam.

# HANDCRAFTED WATER LEVEL IN CANALS. CHANGE WITH MEASURED, IDEALLY.
canal_water_level = 1.2
oWTcanals = [x - canal_water_level for x in surface_canals]


CNM = sparse.csr_matrix(cm)
CNM.eliminate_zeros()  # happens in place. Frees disk usage.


"""
###########################################
Initial configuration of blocks in canals
###########################################
"""
iDamLocation = np.random.randint(0,n_canals,n_blocks).tolist() # Generate random kvector
iWTcanlist = utilities.place_dams(oWTcanals, surface_canals, block_height, iDamLocation, CNM)
wt_canals = iWTcanlist

"""
#########################################
        HYDROLOGY - Parameters
#########################################
"""
ny, nx = dem.shape
dx = 1.; dy = 1. # metres per pixel  
dt = 1. # timestep, in days
diri_bc = 1.2 # ele-phi in meters

hini = - 0.9 # initial wt wrt surface elevation in meters.

# Clean dem... REVIEW
#dem[dem==-9999] = np.nan
#x = np.arange(0, nx)
#y = np.arange(0, ny)
#dem = np.ma.masked_invalid(dem)
#xx, yy = np.meshgrid(x, y)
#x1 = xx[~dem.mask] # get only the valid values
#y1 = yy[~dem.mask]
#clean_dem = dem[~dem.mask] 

ele = dem
Hinitial = ele + hini #initial h (gwl) in the compartment.
#depth = np.ones(shape=Hinitial.shape) # elevation of the impermeable bottom from comon ref. point




# catchment mask
catchment_mask = np.ones(shape=Hinitial.shape, dtype=bool)
catchment_mask[np.where(dem==-9999)] = False # -9999 is current value of dem for outlayer points.


"""
###########################################
    Metropolis/Simulated Annealing.
##########################################
"""
dpm = DryPeatVolumeMinimization(state=iDamLocation, wt_canals=iWTcanlist)
# Useful to make an automatic estimation of parameters:
# - minutes = how much time do you want to wait in minutes.
# - steps = ? Look at source code
#dpm.auto(minutes=2.0, steps = 10)
dpm.steps = 50
dpm.Tmax = 300.0
dpm.Tmin = 1.0
dpm.copy_strategy = "slice" # since the state is just a list, slice is the fastest way to copy
dam_locations, dry_peat_volume = dpm.anneal()

"""
#############################
Final printings and plots
#############################
"""
plt.close('all')
# Call hydrology one last time to plot stuff
wt_can_list = utilities.place_dams(oWTcanals, surface_canals, block_height, dam_locations, CNM)
wt_canal_arr = np.zeros((ny,nx)) # (nx,ny) array with wt canal height in corresponding nodes
for canaln, coords in enumerate(c_to_r_list):
    wt_canal_arr[coords] = wt_can_list[canaln]
    Hinitial[coords] = wt_can_list[canaln]
dry_peat_volume = hydro.hydrology(nx, ny, dx, dy, dt, ele, Hinitial, catchment_mask,
                                  wt_canal_arr, value_for_masked= 0.9, diri_bc=None, neumann_bc = 0.0, plotOpt=True)

with open(r'C:\Users\L1817\Dropbox\PhD\Computation\Indonesia_WaterTable\ForestCarbon\computation\after_21.11.2018\results_simanneal.txt', 'a') as output_file:
    output_file.write("\n dry_peat_volume = " + str(dry_peat_volume) + "    blocked dams:" + str(dam_locations) + "   DATE: " + str(time.ctime()) + "    Montecarlo steps = " + str(dpm.steps))
print("\n dry_peat_volume = ", dry_peat_volume)
timespent = time.time() - time0
utilities.print_time_in_mins(timespent)