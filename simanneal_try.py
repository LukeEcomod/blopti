# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 11:13:41 2018

@author: L1817
"""
from __future__ import print_function
import argparse
import time
import pickle
import numpy as np
import preprocess_data,  utilities, hydro, hydro_utils
import random
from simanneal import Annealer



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
        prohibited_node_list = [i for i,_ in enumerate(oWTcanlist[1:]) if oWTcanlist[1:][i] < self.wt_canals[1:][i]]      # [1:] is to take the 0th element out of the loop
        candidate_node_list = [e for e in range(1, n_canals) if e not in prohibited_node_list] # remove 0 from the range of possible canals
        random.shuffle(candidate_node_list) # Happens in-place.
        dam_to_add = candidate_node_list[0]
        
        # Select dam to remove
        random.shuffle(self.state) # Shuffle to select which canal to remove. Happens in place.
        dam_to_remove = self.state[0]
        
        self.state.remove(dam_to_remove)
        self.state.append(dam_to_add)

    def energy(self):
        self.wt_canals = utilities.place_dams(oWTcanlist, srfcanlist, block_height,
                                self.state, CNM)
        
        wt_canal_arr = np.zeros((ny,nx)) # (nx,ny) array with wt canal height in corresponding nodes
        for canaln, coords in enumerate(c_to_r_list):
            if canaln == 0:
                continue # because c_to_r_list begins at 1
            wt_canal_arr[coords] = self.wt_canals[canaln]
#            Hinitial[coords] = self.wt_canals[canaln] # It should be like this, but not of much use if hydrology is doing the right thing.
        
        e = hydro.hydrology('transient', nx, ny, dx, dy, DAYS, ele, phi_ini, catchment_mask, wt_canal_arr, boundary_arr,
                                                      peat_type_mask=peat_type_mask, httd=h_to_tra_and_C_dict, tra_to_cut=tra_to_cut, sto_to_cut=sto_to_cut,
                                                      diri_bc=diri_bc, neumann_bc = None, plotOpt=False, remove_ponding_water=True)
        return e


"""
Parse command-line arguments
"""
parser = argparse.ArgumentParser(description='Run simulated annealing.')

parser.add_argument('-d','--days', default=3, help='(int) Number of outermost iterations of the fipy solver, be it steadystate or transient. Default=10.', type=int)
parser.add_argument('-b','--nblocks', default=5, help='(int) Number of blocks to locate. Default=5.', type=int)
parser.add_argument('-n','--niter', default=3, help='(int) Number of repetitions of the whole computation. Default=10', type=int)
args = parser.parse_args()

DAYS = args.days
N_BLOCKS = args.nblocks
N_ITER = args.niter

        
"""
Read and preprocess data
"""
retrieve_canalarr_from_pickled = False
preprocessed_datafolder = r"data"
dem_rst_fn = preprocessed_datafolder + r"/lidar_100_resampled_interp.tif"
can_rst_fn = preprocessed_datafolder + r"/canal_clipped_resampled_2.tif"
peat_type_rst_fn = preprocessed_datafolder + r"/Landcover_clipped.tif"
peat_depth_rst_fn = preprocessed_datafolder + r"/peat_depth.tif"


if 'CNM' and 'cr' and 'c_to_r_list' not in globals():
    CNM, cr, c_to_r_list = preprocess_data.gen_can_matrix_and_raster_from_raster(can_rst_fn=can_rst_fn, dem_rst_fn=dem_rst_fn)

elif retrieve_canalarr_from_pickled==True:
    pickle_folder = r"C:\Users\L1817\Winrock"
    pickled_canal_array_fn = r'\DEM_and_canals.pkl'
    with open(pickle_folder + pickled_canal_array_fn) as f:
        CNM, cr, c_to_r_list = pickle.load(f)
    print("Canal adjacency matrix and raster loaded from pickled.")
    
else:
    print("Canal adjacency matrix and raster loaded from memory.")
    
_ , dem, peat_type_arr, peat_depth_arr = preprocess_data.read_preprocess_rasters(can_rst_fn, dem_rst_fn, peat_type_rst_fn, peat_depth_rst_fn)

print("rasters read and preprocessed from file")

# catchment mask
catchment_mask = np.ones(shape=dem.shape, dtype=bool)
catchment_mask[np.where(dem<-10)] = False # -99999.0 is current value of dem for nodata points.

# peel the dem. Only when dem is not surrounded by water
boundary_mask = utilities.peel_raster(dem, catchment_mask)
 
# after peeling, catchment_mask should only be the fruit:
#catchment_mask[boundary_mask] = False

# soil types and soil physical properties and soil depth:
peat_type_mask = peat_type_arr * catchment_mask
peat_bottom_elevation = - peat_depth_arr * catchment_mask # meters with respect to dem surface. Should be negative!


h_to_tra_and_C_dict = hydro_utils.peat_map_interp_functions() # Load peatmap soil types' physical properties dictionary
#soiltypes[soiltypes==255] = 0 # 255 is nodata value. 1 is water (useful for hydrology! Maybe, same treatment as canals).

#BOTTOM_ELE = -6.0 
#peat_bottom_elevation = np.ones(shape=dem.shape) * BOTTOM_ELE
#peat_bottom_elevation = peat_bottom_elevation*catchment_mask
tra_to_cut = hydro_utils.peat_map_h_to_tra(soil_type_mask=peat_type_mask,
                                           gwt=peat_bottom_elevation, h_to_tra_and_C_dict=h_to_tra_and_C_dict)
sto_to_cut = hydro_utils.peat_map_h_to_sto(soil_type_mask=peat_type_mask,
                                           gwt=peat_bottom_elevation, h_to_tra_and_C_dict=h_to_tra_and_C_dict)
sto_to_cut = sto_to_cut * catchment_mask.ravel()

srfcanlist =[dem[coords] for coords in c_to_r_list]

n_canals = len(c_to_r_list)
block_height = 0.4 # water level of canal after placing dam.

# HANDCRAFTED WATER LEVEL IN CANALS. CHANGE WITH MEASURED, IDEALLY.
canal_water_level = 1.2
oWTcanlist = [x - canal_water_level for x in srfcanlist]


"""
###########################################
Initial configuration of blocks in canals
###########################################
"""
iDamLocation = np.random.randint(1,n_canals,N_BLOCKS).tolist() # Generate random kvector
iWTcanlist = utilities.place_dams(oWTcanlist, srfcanlist, block_height, iDamLocation, CNM)
wt_canals = iWTcanlist

"""
#########################################
        HYDROLOGY - Parameters
#########################################
"""
ny, nx = dem.shape
dx = 1.; dy = 1. # metres per pixel  
dt = 1. # timestep, in days
diri_bc = 0.0 # ele-phi in meters

hini = - 0.0 # initial wt wrt surface elevation in meters.

boundary_arr = boundary_mask * (dem - diri_bc) # constant Dirichlet value in the boundaries

ele = dem[:]

phi_ini = ele + hini #initial h (gwl) in the compartment.
phi_ini = phi_ini * catchment_mask


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
dpm.steps = N_ITER
dpm.Tmax = 300.0
dpm.Tmin = 1.0
dpm.copy_strategy = "slice" # since the state is just a list, slice is the fastest way to copy
dam_locations, dry_peat_volume = dpm.anneal()

"""
#############################
Final printings and plots
#############################
"""

# Call hydrology one last time to plot stuff
#wt_can_list = utilities.place_dams(oWTcanals, surface_canals, block_height, dam_locations, CNM)
#wt_canal_arr = np.zeros((ny,nx)) # (nx,ny) array with wt canal height in corresponding nodes
#for canaln, coords in enumerate(c_to_r_list):
#    wt_canal_arr[coords] = wt_can_list[canaln]
#    Hinitial[coords] = wt_can_list[canaln]
#dry_peat_volume = hydro.hydrology(nx, ny, dx, dy, dt, ele, Hinitial, catchment_mask,
#                                  wt_canal_arr, value_for_masked= 0.9, diri_bc=None, neumann_bc = 0.0, plotOpt=True)

with open(r'output/results_sa.txt', 'a') as output_file:
        output_file.write("\n" + str(dry_peat_volume) + "    " + str(N_BLOCKS) + "    " + str(N_ITER) + "    " + str(DAYS) + "    " + str(time.ctime()))
